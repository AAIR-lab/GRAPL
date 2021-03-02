'''
Created on Mar 26, 2020

@author: rkaria
'''

from dulwich import porcelain


def get_head_commit_sha(repo_path):

    return porcelain.Repo(str(repo_path)).head().decode()


def get_active_branch(repo_path):

    return porcelain.active_branch(str(repo_path)).decode()


def is_dirty(repo_path):

    status = porcelain.status(str(repo_path))

    return len(status.staged["add"]) > 0 \
        or len(status.staged["delete"]) > 0 \
        or len(status.staged["modify"]) > 0 \
        or len(status.unstaged) > 0
