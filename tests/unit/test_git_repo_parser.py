"""Git 仓库解析器测试。"""

from pathlib import Path
from uuid import uuid4

import pytest
from git import InvalidGitRepositoryError, Repo

from labflow.parsers.git_repo_parser import GitRepoParser


def create_repo_path(case_name: str) -> Path:
    """在工作区内创建独立测试目录。"""

    repo_path = Path(".tmp") / "tests" / f"{case_name}-{uuid4().hex}"
    repo_path.mkdir(parents=True, exist_ok=True)
    return repo_path


def commit_file(repo: Repo, repo_path: Path, name: str, content: str, message: str) -> None:
    """创建一个提交，方便我验证解析结果。"""

    file_path = repo_path / name
    file_path.write_text(content, encoding="utf-8")
    repo.index.add([name])
    repo.index.commit(message)


def test_git_repo_parser_collects_recent_commits_and_diff() -> None:
    """我会拉出最近提交和当前工作区 diff。"""

    repo_path = create_repo_path("git-parser")
    repo = Repo.init(repo_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "LabFlow Dev")
        config.set_value("user", "email", "labflow@example.com")

    commit_file(repo, repo_path, "notes.txt", "v1\n", "feat: add notes")
    commit_file(repo, repo_path, "notes.txt", "v2\n", "feat: refine notes")
    (repo_path / "notes.txt").write_text("v3\n", encoding="utf-8")

    parser = GitRepoParser()
    result = parser.parse(repo_path)

    assert len(result.recent_commits) == 2
    assert result.recent_commits[0].summary == "feat: refine notes"
    assert "notes.txt" in result.working_tree_diff
    assert result.branch_name


def test_git_repo_parser_rejects_non_repo_directory(monkeypatch) -> None:
    """普通目录不该被我当成仓库。"""

    repo_path = create_repo_path("non-repo")
    parser = GitRepoParser()

    class FailingRepo:
        """我用这个假对象模拟 GitPython 的报错。"""

        def __new__(cls, *args, **kwargs):
            raise InvalidGitRepositoryError("bad repo")

    monkeypatch.setattr("labflow.parsers.git_repo_parser.Repo", FailingRepo)

    with pytest.raises(ValueError, match="不是 Git 仓库"):
        parser.parse(repo_path)
