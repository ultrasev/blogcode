# --------------------------------------------
import argparse
import ast
import functools
import json
import os
import re
import traceback
import typing

import httpx
from bs4 import BeautifulSoup
from pandoranext.hugchat.api import chat

from logger import logger
from prompt import PROMPT


class HtmlPage(object):
    def __init__(self, host: str, id: int) -> None:
        self.url = '{}/{}.html'.format(host, id)
        self._content = None

    def __str__(self) -> str:
        return self.url

    def __bool__(self) -> bool:
        return repr(self) != "" and self._content

    def __repr__(self) -> str:
        if self._content:
            return self._content
        html = httpx.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        content = soup.find('div', class_='entry-content')
        if "option" in str(content):
            return ""
        self._content = content
        return self._content


class ContentGetter(object):
    host = 'http://www.dydhhy.com'

    @classmethod
    def run(cls, start_index: int, end_index: int) -> typing.Generator:
        for i in range(start_index, end_index):
            page = HtmlPage(cls.host, i)
            logger.info("processing: {}".format(page))
            if page:
                with open('data/{}.txt'.format(i), 'w') as file:
                    file.write(repr(page))
                yield i, repr(page)


class FilmInfo(object):
    def __init__(self, raw: str, export_to: str,
                 parser: typing.Callable) -> None:
        """ Extracts film information from raw HTML using a parser.

        Args:
            raw (str): The raw HTML content.
            export_to (str): The location where the parsed results will be saved.
            parser (typing.Callable): The function used to parse film data.
        """
        self.export_to = export_to
        self.raw = raw
        self.generated = None
        self.parser = parser

    def __bool__(self):
        # whether the parsed content has been saved
        return os.path.exists(self.export_to)

    def __enter__(self):
        if not self:
            inputs = PROMPT.format(self.raw)
            self.generated = self.parser(inputs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logger.error({
                "exc_type": exc_type,
                "exc_value": exc_value,
                "traceback": traceback
            })

    @property
    def json(self) -> typing.Dict:
        assert self.generated, "No parsed content found"
        content = re.search(r'(\{.*?\})', self.generated, re.DOTALL)
        if content:
            try:
                return json.loads(content.group(1))
            except json.decoder.JSONDecodeError:
                try:
                    return ast.literal_eval(content.group(1))
                except SyntaxError as e:
                    logger.error({
                        "Error": "SyntaxError: {}".format(e),
                        "generated": self.generated,
                        "traceback": traceback.format_exc(),
                        "msg": "Cannot parse content"
                    })
                    return {"text": self.generated}
        return {"text": self.generated}

    def dump(self) -> None:
        with open(self.export_to, 'w') as file:
            json.dump(self.json, file, indent=4, ensure_ascii=False)

    def __str__(self) -> str:
        return self.export_to


class FilmInfoExtractor(object):
    def __init__(self, export_path: str) -> None:
        """ Extracts film information from raw HTML using a parser.

        Args:
            parser (typing.Callable): The function used to parse film data.
            export_to (str): The location where the parsed results will be saved.
        """
        self.export_path = export_path
        self.raw_path = 'data/raw'
        if not os.path.exists(export_path):
            os.makedirs(export_path)

    def __str__(self):
        return str({
            "class": "FilmInfoExtractor",
            "export_path": self.export_path,
        })

    def __iter__(self) -> typing.Generator:
        for f in os.listdir(self.raw_path):
            if f.endswith('.txt'):
                with open(os.path.join(self.raw_path, f), 'r') as file:
                    yield f, file.read()

    def parse_with(self, parser: typing.Callable) -> None:
        processed = 0
        for fpath, html in self:
            export_to = os.path.join(self.export_path,
                                     fpath.replace('.txt', '.json'))
            processed += 1
            with FilmInfo(raw=html,
                          export_to=export_to,
                          parser=parser) as filminfo:
                if filminfo:
                    logger.info({
                        "processed": processed,
                        "msg": "skipping {}".format(filminfo)
                    })
                else:
                    filminfo.dump()
                    logger.info({
                        "processed": processed,
                        "msg": "processed {}".format(filminfo)
                    })


class ExperimentRunner(object):
    result_path = 'results'
    models = {
        "dpo": "Mixtral-8X7B-DPO",
        "mixtral-8x7b-instruct": "Mixtral-8X7B-Instruct",
        "llama-2": "Llama-2-70B-chat",
        "codellama": "CodeLlama-70b-Instruct",
        "openchat": "openchat-3.5",
        "gemma": "Gemma-7B",
        "mistral": "Mistral-7B-Instruct"
    }

    def run(self):
        for model in self.models:
            parser = functools.partial(chat, model=model, stream=True)
            for _ in range(3):
                try:
                    export_path = os.path.join(self.result_path, model)
                    extractor = FilmInfoExtractor(export_path)
                    logger.info("running experiment for {}".format(extractor))
                    extractor.parse_with(parser)
                    break
                except Exception as e:
                    logger.error({
                        "error": e,
                        "traceback": traceback.format_exc()
                    })


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--test", action="store_true")
    args = argparser.parse_args()
    if args.test:
        ExperimentRunner().run()
