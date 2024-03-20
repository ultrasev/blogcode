

import re
import json
import os


class A(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        return self[name]


path = 'results/dpo'

for f in os.listdir(path):
    if f.endswith(".json"):
        with open(f'{path}/{f}', 'r') as file:
            data = json.load(file)
            if 'text' in data:
                os.remove(f'{path}/{f}')
                continue
            name = data['电影名称']
            if not re.search('[\u4e00-\u9fff]', name):
                print(f)
                os.remove(f'{path}/{f}')
