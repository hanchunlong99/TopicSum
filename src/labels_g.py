# _*_ coding: utf-8 _*_
# @Time: 2022/10/8 2:43 
# @Author: 韩春龙
from src.data_preprocessing.MemSum.utils import greedy_extract
import json

with open("../data/pubmed/val_PUBMED.jsonl","r") as f:
    for line in f:
        break
example_data = json.loads(line)
print(example_data.keys())
greedy_extract(example_data["text"], example_data["summary"], beamsearch_size=1)[0]
