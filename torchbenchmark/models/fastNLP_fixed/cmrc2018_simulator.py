"""
Generator of a simulated CMRC2018 dataset.
Use random Chinese characters with the same length as the original dataset.
"""
import os
import pathlib
import numpy
import json
import random
import patch
import fastNLP

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

CMRC2018_TRAIN_SPEC = {
    # Original
    # "data_size": 2403,
    # Benchmark
    "data_size": TRAIN_BATCH_SIZE,
    "title_length": 5,
    "paragraph_size": 1,
    "context_length": 456,
    "qas_size": 5,
    "query_length": 15,
    "answers_size": 1,
    "answers_length": 7
}
CMRC2018_DEV_SPEC = {
    # Original
    # "data_size": 848,
    # Benchmark
    "data_size": EVAL_BATCH_SIZE,
    "title_length": 4,
    "paragraph_size": 1,
    "context_length": 455,
    "qas_size": 4,
    "query_length": 15,
    "answers_size": 3,
    "answers_length": 7
}

CMRC2018_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "cmrc2018-sim")
CMRC2018_CONFIG_DIR = os.path.join(CMRC2018_DIR, "config")
CMRC2018_TRAIN_SIM = os.path.join(CMRC2018_DIR, "train.json")
CMRC2018_DEV_SIM = os.path.join(CMRC2018_DIR, "dev.json")
CMRC2018_VOCAB_SIM = os.path.join(CMRC2018_CONFIG_DIR, "vocab.txt")
CMRC2018_BERT_CONFIG = os.path.join(CMRC2018_CONFIG_DIR, "bert_config.json")
VOCAB_SET = set()

# Generate random Chinese string with length l
def _GBK2312(l):
    head = 0xd7
    while head == 0xd7:
        head = random.randint(0xb0, 0xf7)
    body = random.randint(0xa1, 0xfe)
    val = f'{head:x} {body:x}'
    s = bytes.fromhex(val).decode('gb2312')
    VOCAB_SET.add(s)
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

def _generate_dev():
    dev_data = _generate_cmrc2018(CMRC2018_DEV_SPEC)
    _dump_data(dev_data, CMRC2018_DEV_SIM)

def _generate_train():
    dev_data = _generate_cmrc2018(CMRC2018_TRAIN_SPEC)
    _dump_data(dev_data, CMRC2018_TRAIN_SIM)

# MUST be called after generate_dev() AND generate_train()!
def _generate_vocab():
    never_split = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    VOCAB_SET.update(never_split)
    with open(CMRC2018_VOCAB_SIM, "w") as vf:
        vf.write("\n".join(list(VOCAB_SET)))

def _copy_bert_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "bert_config.json"), "r") as configf:
        config = configf.read()
    with open(CMRC2018_BERT_CONFIG, "w") as configf:
        configf.write(config)

def _setup_os_env():
    os.environ["TORCHBENCH_FASTNLP_CONFIG_PATH"] = CMRC2018_BERT_CONFIG

def _create_empty_bin():
    CMRC2018_CONFIG_DIR = os.path.join(CMRC2018_DIR, "config")
    bin_file = os.path.join(CMRC2018_CONFIG_DIR, "chinese_wwm_pytorch.bin")
    with open(bin_file, "w") as bf:
        bf.write("")

def try_patch_fastnlp():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "fastnlp.patch")
    fastNLP_dir = os.path.dirname(fastNLP.__file__)
    fastNLP_target_file = os.path.join(fastNLP_dir, "embeddings", "bert_embedding.py")
    p = patch.fromfile(patch_file)
    return p.apply(strip=1, root=fastNLP_dir)

def generate_inputs():
    _create_dir_if_nonexist(CMRC2018_DIR)
    _create_dir_if_nonexist(os.path.join(CMRC2018_DIR, "config"))
    _generate_dev()
    _generate_train()
    _generate_vocab()
    _create_empty_bin()
    _copy_bert_config()
    _setup_os_env()
