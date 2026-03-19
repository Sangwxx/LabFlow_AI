"""Git 仓库解析器测试。"""

from pathlib import Path
from uuid import uuid4

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


def test_git_repo_parser_supports_plain_source_directory(monkeypatch) -> None:
    """即使目录里没有 .git，我也要把它当成待分析源码目录。"""

    repo_path = create_repo_path("non-repo")
    (repo_path / "trainer.py").write_text("alpha = 0.30\nbeta = 0.70\n", encoding="utf-8")

    class FailingRepo:
        """我用这个假对象强制走普通目录兜底分支。"""

        def __new__(cls, *args, **kwargs):
            raise InvalidGitRepositoryError("bad repo")

    monkeypatch.setattr("labflow.parsers.git_repo_parser.Repo", FailingRepo)

    result = GitRepoParser().parse(repo_path)

    assert result.source_type == "directory"
    assert result.branch_name == "UNVERSIONED"
    assert result.recent_commits == ()
    assert "diff --git a/trainer.py b/trainer.py" in result.working_tree_diff
    assert result.source_files[0].relative_path == "trainer.py"
