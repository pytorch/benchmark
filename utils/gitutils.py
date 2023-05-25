"""gitutils.py

Utils for getting git-related information.
"""

import git
import re
import os
import time
import subprocess
from datetime import datetime
from typing import Optional, List

# Assume the nightly branch commit message is in the following format
# Hash in the parentheses links to the commit on the master branch
NIGHTLY_COMMIT_MSG = "nightly release \((.*)\)"


def get_torch_main_commit(pytorch_repo: str, nightly_commit: str):
    repo = git.Repo(pytorch_repo)
    msg = repo.commit(nightly_commit).message
    # There are two possibilities of the hash `nightly_commit`:
    # 1. The hash belongs to the nightly branch
    #    If so, the git commit message should match `NIGHTLY_COMMIT_MSG`
    # 2. The hash belongs to the master/main branch
    #    We can directly use this hash in this case
    nightly_commit_regex = re.compile(NIGHTLY_COMMIT_MSG)
    search_result = nightly_commit_regex.search(msg)
    if search_result:
        return search_result.group(1)
    # We now believe the commit now belongs to the master/main branch
    # Unfortunately, there is no way to map a commit back to a branch with gitpython
    return nightly_commit

def clean_git_repo(repo: str) -> bool:
    try:
        command = f"git clean -xdf"
        subprocess.check_call(command, cwd=repo, shell=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to cleanup git repo {repo}")
        return None

def update_git_repo_branch(repo: str, branch: str) -> bool:
    try:
        command = f"git pull origin {branch}"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to update git repo {repo}, branch {branch}")
        return None

def get_git_commit_on_date(repo: str, date: datetime) -> Optional[str]:
    try:
        # Get the first commit since date
        formatted_date = date.strftime("%Y-%m-%d")
        command = f"git log --until={formatted_date} -1 --oneline | cut -d ' ' -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get the last commit on date {formatted_date} in repo {repo}")
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
        if out == ['']:
            out = None
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

def cleanup_local_changes(repo: str):
    command = ["git", "reset", "--hard", "HEAD"]
    subprocess.check_call(command, cwd=repo, shell=False)

def checkout_git_commit(repo: str, commit: str) -> bool:
    try:
        assert len(commit) != 0
        cleanup_local_changes(repo)
        command = ["git", "checkout", "--recurse-submodules", commit]
        subprocess.check_call(command, cwd=repo, shell=False)
        return True
    except subprocess.CalledProcessError:
        # Sleep 5 seconds for concurrent git process, remove the index.lock file if exists, and try again
        try:
            time.sleep(5)
            index_lock = os.path.join(repo, ".git", "index.lock")
            if os.path.exists(index_lock):
                os.remove(index_lock)
            command = ["git", "checkout", "--recurse-submodules", commit]
            subprocess.check_call(command, cwd=repo, shell=False)
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to checkout commit {commit} in repo {repo}")
            return False

def update_git_repo(repo: str, branch: str="main") -> bool:
    try:
        print(f"======================= [TORCHBENCH] Updating repository {repo} branch {branch} =======================")
        assert len(branch) != 0
        command = ["git", "checkout", "--recurse-submodules", branch]
        subprocess.check_call(command, cwd=repo, shell=False)
        command = ["git", "pull"]
        subprocess.check_call(command, cwd=repo, shell=False)
        command = ["git", "checkout", "--recurse-submodules", branch]
        subprocess.check_call(command, cwd=repo, shell=False)
        return True
    except subprocess.CalledProcessError:
        # Sleep 5 seconds for concurrent git process, remove the index.lock file if exists, and try again
        try:
            time.sleep(5)
            print(f"======================= [TORCHBENCH] Updating repository {repo} branch {branch} (2nd try) =======================")
            index_lock = os.path.join(repo, ".git", "index.lock")
            if os.path.exists(index_lock):
                os.remove(index_lock)
            command = ["git", "checkout", "--recurse-submodules", branch]
            subprocess.check_call(command, cwd=repo, shell=False)
            command = ["git", "pull"]
            subprocess.check_call(command, cwd=repo, shell=False)
            command = ["git", "checkout", "--recurse-submodules", branch]
            subprocess.check_call(command, cwd=repo, shell=False)
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to update to branch {branch} in repo {repo}")
            return False
