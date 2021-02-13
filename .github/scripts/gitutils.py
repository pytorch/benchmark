"""gitutils.py

Utils for getting git-related information.
"""

import os
from typing import Optional, List
import subprocess

def get_git_origin(repo: str):
    pass


def get_git_commits(repo: str, start: str, end: str) -> Optional[List[str]]:
    try:
        command = f"git log --reverse --oneline --ancestry-path {start}^..{end} | cut -d \" \" -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip().split("\n")
        return out
    except subprocess.CalledProcessError:
        print(f"git command {command} returns non-zero status.")
        return None

def get_current_commit(repo: str) -> Optional[str]:
    try:
        command = f"git log --reverse --oneline -1 | cut -d \" \" -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get the current commit in repo {repo}")
        return None

def checkout_git_commit(repo: str, commit: str) -> bool:
    try:
        print(f"Checking out commit {commit}...", end="", flush=True)
        command = f"git checkout {commit} &> /dev/null"
        subprocess.run(command, cwd=repo, shell=True)
        command = f"git submodule sync &> /dev/null && git submodule update --init --recursive &> /dev/null"
        subprocess.run(command, cwd=repo, shell=True)
        print("done")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to checkout commit f{commit} in repo {repo}")
        return False

def update_from_remote(repo: str) -> bool:
    if not checkout_git_commit(repo, "master"):
        return False
    try:
        
def test_get_git_commits():
    repo = os.path.expandvars("${HOME}/pytorch")
    start = "94e328c"
    end = "65876d3"
    result = get_git_commits(repo, start, end)
    answer = ["94e328c038", "8954eb3f72", "a9137aeb06", "40d7c1091f"]
    assert result, "Can't get git commit history!"
    assert result[:4] == answer

def test_checkout_commit():
    repo = os.path.expandvars("${HOME}/pytorch")
    commit1 = "65876d3" 
    assert checkout_git_commit(repo, commit1)
    assert (get_current_commit(repo) == "65876d3f51")
    assert checkout_git_commit(repo, "master")

if __name__ == "__main__":
    # test_get_git_commits()
    test_checkout_commit()
    
