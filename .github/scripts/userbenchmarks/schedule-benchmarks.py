import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent

def list_all_userbenchmarks():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["gcp-a100"], required=True, help="Specify the benchmark platform.")
    args = parser.parse_args()
