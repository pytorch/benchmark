import json
import os

from typing import Dict

GITHUB_ISSUE_TEMPLATE = """
TorchBench CI has detected a performance signal or runtime regression, and bisected its result.

Control PyTorch commit: {control_commit}
Control PyTorch version: {control_version}

Treatment PyTorch commit: {treatment_commit}
Treatment PyTorch version: {treatment_version}

Bisection result:

```
{result}
```

cc {owner}
"""

DEFAULT_GH_ISSUE_OWNER = "@xuzhao9"

def process_bisection_into_gh_issue(bisection_output_json: str, output_path: str) -> None:
    with open(bisection_output_json, "r") as fp:
        bisection = json.load(fp)

    result = json.dump(bisection, indent=4)
    control_commit = bisection["start"]
    control_version = bisection["start_version"]
    treatment_commit = bisection["end"]
    treatment_version = bisection["end_version"]

    if "GITHUB_ENV" in os.environ:
        fname = os.environ["GITHUB_ENV"]
        content = f"TORCHBENCH_BISECTION_COMMIT_FOUND_OR_FAILED='{bisection.target_repo.end}'\n"
        with open(fname, 'a') as fo:
            fo.write(content)
        process_bisection_into_gh_issue(bisection.output_json)

    github_run_id = os.environ.get("GITHUB_RUN_ID", None)
    github_run_url = "No URL found, please look for the failing action in " + \
                     "https://github.com/pytorch/benchmark/actions"
    if github_run_id is not None:
        github_run_url = f"https://github.com/pytorch/benchmark/actions/runs/{github_run_id}"

    issue_config: Dict[str, str] = {
        "control_commit": control_commit,
        "treatment_commit": treatment_commit,
        "control_version": control_version,
        "treatment_version": treatment_version,
        "result": result,
        "github_run_url": github_run_url,
        "owner": DEFAULT_GH_ISSUE_OWNER
    }

    issue_body = GITHUB_ISSUE_TEMPLATE.format(**issue_config)
    with open(output_path, "w") as f:
        f.write(issue_body)
