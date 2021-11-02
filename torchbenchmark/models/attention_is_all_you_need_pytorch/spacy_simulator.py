""" Analyze and simulate the spacy multi30k dataset.
Generate random English/German texts with the same vocab frequency as the original dataset.
"""
import os
import io
import ast
import yaml
import random
import requests
import argparse
import zufallsworte as zufall
from contextlib import redirect_stdout

LANG_LIST = ["de", "en"]
DATA_FILES = ["train", "val", "test2016"]
WORD_SITE = "https://www.mit.edu/~ecprice/wordlist.10000"
random.seed(1337)

def analyze_dataset(dataset_dir, lang):
    # Check if the dataset files exist
    files = list(map(lambda x: f"{x}.{lang}", DATA_FILES))
    dataset_meta = {}
    dataset_meta["files"] = {}
    for f in files:
        fpath = os.path.join(dataset_dir, f)
        assert os.path.exists(fpath), f"Missing multi30k data file: {fpath}"
        dataset_meta["files"][f] = {}
        dataset_meta["files"][f]["distribution"] = {}
        with open(fpath, "r") as datafile:
            data = datafile.read()
        for line in data.splitlines():
            line = line.strip()
            words_len = len(list(filter(lambda x: x.strip(), line.split(' '))))
            if not words_len:
                continue
            if words_len not in dataset_meta["files"][f]["distribution"]:
                dataset_meta["files"][f]["distribution"][words_len] = 0
            dataset_meta["files"][f]["distribution"][words_len] += 1
        dataset_meta["files"][f]["size"] = len(data.splitlines())
        dataset_meta["lang"] = lang
    return dataset_meta

def generate_random_sentence(words, length, lang):
    assert lang == "en" or lang == "de", "Only English and German dataset is supported"
    if lang == "en":
        selected_words = random.sample(range(len(words)), length)
        sentence = []
        for word_index in selected_words:
            sentence.append(words[word_index])
        return sentence
    elif lang == "de":
        f = io.StringIO()
        with redirect_stdout(f):
            zufall.zufallswoerter(length)
        out = f.getvalue()
        return ast.literal_eval(out)

def generate_dataset(output_dir, config):
    WORDS = []
    if config["lang"] == "en":
        response = requests.get(WORD_SITE)
        WORDS = response.text.splitlines()
    for f in config["files"]:
        dataset = []
        for sentence_len in config["files"][f]["distribution"]:
            slen = sentence_len
            for _ in range(config["files"][f]["distribution"][slen]):
                s = generate_random_sentence(WORDS, slen, config["lang"])
                dataset.append(" ".join(s) + "\n")
        with open(os.path.join(output_dir, f), "w") as outfile:
            outfile.write(''.join(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--data-dir', required=True)
    analyze_parser.add_argument('--output-dir', required=True)
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('--config', required=True)
    generate_parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.command == 'analyze':
        for lang in LANG_LIST:
            config_name = f'multi30k-{lang}-config.yaml'
            config_path = os.path.join(args.output_dir, config_name)
            dataset_meta = analyze_dataset(args.data_dir, lang)
            with open(config_path, "w") as out_file:
                yaml.dump(dataset_meta, out_file)
    elif args.command == 'generate':
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        generate_dataset(args.output_dir, config)
