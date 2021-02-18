"""gitutils.py

Utils for getting git-related information.
"""

import os
from typing import Optional, List
import subprocess

def update_git_repo(repo: str, branch: str) -> bool:
    try:
        command = f"git pull origin {branch}"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to update git repo {repo}, branch {branch}")
        return None

def check_git_exist_local_branch(repo: str, branch: str) -> bool:
    command = f"git rev-parse --verify {branch} &> /dev/null "
    retcode = subprocess.call(command, cwd=repo, shell=True)
    return (retcode == 0)

def get_git_commit_date(repo: str, commit: str) -> str:
    try:
        command = f"git show -s --format=%ci {commit}"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get date of commit {commit} in repo {repo}")
        return None
    
def checkout_git_branch(repo: str, branch: str) -> bool:
    try:
        if check_git_exist_local_branch(repo, branch):
            command = f"git checkout {branch} &> /dev/null "
        else:
            command = f"git checkout --track origin/{branch} &> /dev/null"
        retcode = subprocess.call(command, cwd=repo, shell=True)
        return (retcode == 0)
    except subprocess.CalledProcessError:
        print(f"Failed to checkout git repo {repo}, branch {branch}")
        return None

def get_current_branch(repo: str) -> Optional[str]:
    try:
        command = "git branch --show-current"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get current branch name for repo {repo}")
        return None

def get_git_origin(repo: str) -> Optional[str]:
    try:
        command = "git remote get-url origin"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except:
        print(f"git command {command} returns non-zero status in repo {repo}")
        return None

def get_git_commits(repo: str, start: str, end: str) -> Optional[List[str]]:
    try:
        command = f"git log --reverse --oneline --ancestry-path {start}^..{end} | cut -d \" \" -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip().split("\n")
        return out
    except subprocess.CalledProcessError:
        print(f"git command {command} returns non-zero status in repo {repo}")
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
        assert len(commit) != 0
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

def update_git_repo(repo: str) -> bool:
    try:
        command = "git checkout master && git pull origin master"
        subprocess.run(command, cwd=repo, shell=True)
        command = f"git submodule sync &> /dev/null && git submodule update --init --recursive &> /dev/null"
        subprocess.run(command, cwd=repo, shell=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to update git repo {repo}")
        return False

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
    # test_checkout_commit()
    pass
