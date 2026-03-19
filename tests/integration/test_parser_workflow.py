"""感知层解析流程测试。"""

from pathlib import Path
from uuid import uuid4

from git import Repo

from labflow.parsers.git_repo_parser import GitRepoParser


def test_git_repo_parser_supports_nested_path_input() -> None:
    """即使我给的是仓库子目录，解析器也应该能找到仓库根。"""

    repo_root = Path(".tmp") / "tests" / f"nested-repo-{uuid4().hex}"
    nested_path = repo_root / "src" / "module"
    nested_path.mkdir(parents=True, exist_ok=True)

    repo = Repo.init(repo_root)
    with repo.config_writer() as config:
        config.set_value("user", "name", "LabFlow Dev")
        config.set_value("user", "email", "labflow@example.com")

    file_path = repo_root / "README.md"
    file_path.write_text("demo\n", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("docs: add readme")

    result = GitRepoParser().parse(nested_path)

    assert result.repo_path == str(repo_root.resolve())
    assert result.recent_commits[0].summary == "docs: add readme"
