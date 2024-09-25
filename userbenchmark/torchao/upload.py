import argparse
import csv
import os
import subprocess
import warnings
from pathlib import Path
from typing import List


def _get_torchao_head_sha():
    cmd_args = ["git", "ls-remote", "https://github.com/pytorch/ao.git", "HEAD"]
    sha = subprocess.check_output(cmd_args).decode().split("\t")[0]
    return sha


def _get_model_set(filename: str):
    if "timm_models" in filename:
        return "timm"
    if "huggingface" in filename:
        return "huggingface"
    if "torchbench" in filename:
        return "torchbench"
    raise RuntimeError(f"Unknown model set from filename: {filename}")


def post_ci_process(output_files: List[str]):
    for path in output_files:
        perf_stats = []
        path = Path(path).absolute()
        modelset = _get_model_set(path.name)
        test_name = f"torchao_{modelset}_perf"
        runner = "gcp_a100"
        job_id = 0
        workflow_run_id = os.environ.get("WORKFLOW_RUN_ID", 0)
        workflow_run_attempt = os.environ.get("WORKFLOW_RUN_ATTEMPT", 0)
        filename = os.path.splitext(os.path.basename(path))[0]
        head_repo = "pytorch/ao"
        head_branch = "main"
        head_sha = _get_torchao_head_sha()
        print(f"Processing file {path} ...")
        # When the test fails to run or crashes, the output file does not exist.
        if not path.exists():
            warnings.warn(f"Expected output file {path} does not exist.")
            continue
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")

            for row in reader:
                row.update(
                    {
                        "workflow_id": workflow_run_id,  # type: ignore[dict-item]
                        "run_attempt": workflow_run_attempt,  # type: ignore[dict-item]
                        "test_name": test_name,
                        "runner": runner,
                        "job_id": job_id,
                        "filename": filename,
                        "head_repo": head_repo,
                        "head_branch": head_branch,
                        "head_sha": head_sha,
                    }
                )
                perf_stats.append(row)

        # Write the decorated CSV file
        with open(path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=None)

            for i, row in enumerate(perf_stats):
                if i == 0:
                    writer.fieldnames = row.keys()
                    writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-files", nargs="+", help="Add files to test.")
    args = parser.parse_args()
    post_ci_process(args.test_files)
