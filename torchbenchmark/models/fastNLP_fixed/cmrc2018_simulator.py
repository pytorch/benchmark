"""
Generator of a simulated CMRC2018 dataset.
Use random Chinese characters with the same length as the original dataset.
"""
import os
import pathlib
import numpy
import json
import random

CMRC2018_DEV_SPEC = {
    # Original
    # "data_size": 848,
    "data_size": 2,
    "title_length": 4,
    "paragraph_size": 1,
    "context_length": 455,
    "qas_size": 4,
    "query_length": 15, 
    "answers_size": 3,
    "answers_length": 7
}
CMRC2018_TRAIN_SPEC = {
    # Original
    # "data_size": 2403,
    "data_size": 2,
    "title_length": 5,
    "paragraph_size": 1,
    "context_length": 456,
    "qas_size": 5,
    "query_length": 15,
    "answers_size": 1,
    "answers_length": 7
}

CMRC2018_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "cmrc2018-sim")
CMRC2018_DEV_SIM = os.path.join(CMRC2018_DIR, "dev.json")
CMRC2018_TRAIN_SIM = os.path.join(CMRC2018_DIR, "train.json")

# Generate random Chinese string with length l
def _GBK2312(l):
    head = 0xd7
    while head == 0xd7:
        head = random.randint(0xb0, 0xf7)
    body = random.randint(0xa1, 0xfe)
    val = f'{head:x} {body:x}'
    s = bytes.fromhex(val).decode('gb2312')
    if l == 0:
        return s
    else:
        return s + _GBK2312(l-1)

def _generate_cmrc2018(spec):
    simdata = {}
    simdata["version"] = "v1.0-sim"
    simdata["data"] = []
    for ind in range(spec["data_size"]):
        item = {} 
        para = {}
        item["id"] = f"DEV_{ind}"
        item["title"] = _GBK2312(spec["title_length"])
        item["paragraphs"] = []
        para["id"] = item["id"]
        para["context"] = _GBK2312(spec["context_length"])
        para["qas"] = []
        for qind in range(spec["qas_size"]):
            q = {}
            q["question"] = _GBK2312(spec["query_length"])
            q["id"] = f"{item['id']}_QUERY_{qind}"
            q["answers"] = []
            for ans in range(spec["answers_size"]):
                ans = {}
                ans["text"] = _GBK2312(spec["answers_length"])
                ans["answer_start"] = 0
                q["answers"].append(ans)
            para["qas"].append(q)
        item["paragraphs"].append(para)
        simdata["data"].append(item)
    return simdata

def _create_dir_if_nonexist(dirpath):
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)

def _dump_data(data, path):
    with open(path, "w") as dp:
        json.dump(data, dp, indent=4, ensure_ascii=False)

def generate_dev():
    _create_dir_if_nonexist(CMRC2018_DIR)
    dev_data = _generate_cmrc2018(CMRC2018_DEV_SPEC)
    _dump_data(dev_data, CMRC2018_DEV_SIM)

def generate_train():
    _create_dir_if_nonexist(CMRC2018_DIR)
    dev_data = _generate_cmrc2018(CMRC2018_TRAIN_SPEC)
    _dump_data(dev_data, CMRC2018_TRAIN_SIM)
