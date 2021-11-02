""" Analyze and simulate the spacy multi30k dataset.
Generate random English/German texts with the same vocab frequency as the original dataset.
"""
import argparse

LANG_LIST = ["de", "en"]
DATA_FILES = ["train", "val", "test2016"]

# Eng vocab
def analyze_dataset(dataset_dir):
    # Check if the dataset files exist
    for lang in LANG_LIST:
        for f in DATA_FILES:
            fpath = os.path.join(f"{f}.{lang}")

    for lang in LANG_LIST:
        for f in DATA_FILES:
    # key: vocab, val: frequency
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--data-dir', required=True)
    analyze_parser.add_argument('--output-dir', required=True)
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('--config', required=True)
    generate_parser.add_argument('--output-dir', required=True)
    args = parser.parse_arguments()
