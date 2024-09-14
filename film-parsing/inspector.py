#!/usr/bin/env python

import json
import os
import typing

import pydantic

from logger import logger


class LLM(pydantic.BaseModel):
    abbrev: str
    model: str


class Models(object):
    GOLDEN = "gpt-4"

    def __init__(self) -> None:
        self._list = {
            "dpo": "Mixtral-8X7B-DPO",
            "mixtral-8x7b-instruct": "Mixtral-8X7B-Instruct",
            "llama-2": "Llama-2-70B-chat",
            "codellama": "CodeLlama-70b-Instruct",
            "openchat": "openchat-3.5",
            "gemma": "Gemma-7B",
            "mistral": "Mistral-7B-Instruct",
            "juchat-deepseek": "juchat-deepseek"
        }

    def __iter__(self) -> typing.Generator[LLM, None, None]:
        for k, v in self._list.items():
            yield LLM(abbrev=k, model=v)

    def __contains__(self, key: str) -> bool:
        return key in self._list


class FilmItem(pydantic.BaseModel):
    id: int
    content: typing.Dict
    model: str

    def __repr__(self) -> str:
        return self.content

    def __getitem__(self, name: str) -> typing.Any:
        return self.content.get(name, None)

    def __iter__(self) -> typing.Generator:
        for k, v in self.content.items():
            if k != "text":
                yield k, v

    def __bool__(self):
        if "text" in self.content:
            del self.content["text"]
        return bool(self.content)

    def __contains__(self, key):
        return key in self.content


class FilmItemLoader(pydantic.BaseModel):
    model: str

    def __path(self) -> str:
        if self.model in Models():
            return f"results/{self.model}"
        elif self.model.lower() == Models.GOLDEN:
            return "films/parsed"
        raise ValueError("Model {} not found".format(self.model))

    def __iter__(self) -> typing.Generator[FilmItem, None, None]:
        path = self.__path()
        for f in os.listdir(path):
            if f.endswith(".json"):
                fpath = os.path.join(path, f)
                with open(fpath, "r") as file:
                    try:
                        yield FilmItem(id=int(f.split(".")[0]),
                                       content=json.load(file),
                                       model=self.model)
                    except json.decoder.JSONDecodeError:
                        logger.error({
                            "Error": "JSONDecodeError",
                            "file path": fpath,
                            "msg": "Cannot parse content"
                        })


class Score(pydantic.BaseModel):
    llm: LLM
    score: float

    def __str__(self) -> str:
        return "{:25s}: {:.4f}".format(self.llm.abbrev, self.score)


class Inspector(object):
    def __init__(self, llm: LLM):
        self.llm = llm

    def __iter__(
        self
    ) -> typing.Generator[typing.Tuple[FilmItem, FilmItem], None, None]:
        yield from zip(FilmItemLoader(model=Models.GOLDEN),
                       FilmItemLoader(model=self.llm.abbrev))

    def score(self) -> Score:
        """" 对 predictions 中每个电影的结果进行评分。
        1. 格式对齐：如果结果中包含合法的 `json`， +10 分； 不然，直接返回 0 分。
        2. 字段对齐：结果中包含`电影名称`、`上映时间`、`类型`、`简介`、`豆瓣评分`、`豆瓣链接`、`IMDb评分`、`图片链接`、`类型`、`制片国家`、`集数` 等字段，每个字段加 1 分。
        3. 字段对齐：结果中包含额外的字段， -1 分。
        4. 提取内容对齐：`电影名称`、`上映时间`、`豆瓣评分`、`豆瓣链接`、`IMDb评分`、`图片链接`、`制片国家`、`集数` 每个字段的值与 `golden` 一致， 则 +1 分。
        """
        scores, total = [], 29
        for pred, gold in self:
            try:
                assert pred.id == gold.id, "ID mismatched!"
            except AssertionError as e:
                logger.error({
                    "Error": "ID mismatched",
                    "msg": str(e)
                })
                return Score(llm=self.llm, score=0)
            if not pred:
                scores.append(0)
            else:
                score = 10
                score += sum([1 if k in gold else -1 for k, _ in pred])
                keys = ["电影名称", "上映时间", "豆瓣评分", "豆瓣链接",
                        "IMDb评分", "图片链接", "制片国家", "集数"]
                score += len([k for k in keys if pred[k] == gold[k]])
                scores.append(score / total)
        return Score(llm=self.llm, score=max(0, sum(scores) / max(1, len(scores))))


class MarkDown(object):
    def __init__(self, scores: typing.List[Score]) -> None:
        self.scores = scores

    def __str__(self) -> str:
        header = "| Model | Score |"
        self.scores.sort(key=lambda x: x.score, reverse=True)
        body = "\n".join([
            "| {:25s} | {:.4f} |".format(score.llm.model, score.score)
            for score in self.scores
        ])

        return "\n".join([header, "|---|---|", body])


def main():
    scores = []
    for llm in Models():
        score = Inspector(llm).score()
        scores.append(score)
    print(MarkDown(scores))


if __name__ == "__main__":
    main()
