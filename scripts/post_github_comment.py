import argparse
from github import Github
import os

HEADER = "Performance Bot:\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gh_token", required=True, help='github auth token')
    parser.add_argument("--gh_repo", required=True, help='name of repo to interact with')
    parser.add_argument("--gh_org", required=True, help='name of org containing the repo')
    parser.add_argument("--pr_url", help="PR URL")
    parser.add_argument("body_file", help="file/stream from which to read body text")
    args = parser.parse_args()

    g = Github(args.gh_token) 
    repo = g.get_repo(f"{args.gh_org}/{args.gh_repo}")
    pr_num = int(os.path.split(os.getenv('CIRCLE_PULL_REQUEST'))[1]) 
    pr = repo.get_pull(pr_num)

    with open(args.body_file) as f:
        body = f.read()
    
    new_body = HEADER + body
    comment_id = None
    for comment in pr.get_issue_comments():
        if comment.body.find(HEADER) == 0:
            comment_id = comment.id
            print("Editing existing comment in place")
            comment.edit(new_body)

    if comment_id is None:
        print("Creating new comment")
        comment_id = pr.create_issue_comment(new_body) 
