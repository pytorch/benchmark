import argparse
import argparse
from pathlib import Path

def generate_index_file(metrics_path):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True, help="Directory of the metrics.")