"""Git 仓库解析器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from git import InvalidGitRepositoryError, NoSuchPathError, Repo


@dataclass(frozen=True)
class CommitInfo:
    """提交摘要。"""

    hexsha: str
    short_sha: str
    author_name: str
    authored_at: str
    summary: str


@dataclass(frozen=True)
class GitRepoParseResult:
    """Git 仓库解析结果。"""

    repo_path: str
    branch_name: str
    recent_commits: tuple[CommitInfo, ...]
    working_tree_diff: str


class GitRepoParser:
    """基于 GitPython 的本地仓库解析器。"""

    def parse(self, repo_path: str | Path) -> GitRepoParseResult:
        """解析本地 Git 仓库的近期提交和工作区 diff。"""

        path = Path(repo_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Git 路径不存在: {path}")
        if not path.is_dir():
            raise ValueError(f"Git 路径不是目录: {path}")

        try:
            repo = Repo(path, search_parent_directories=True)
        except NoSuchPathError as exc:
            raise FileNotFoundError(f"Git 路径不存在: {path}") from exc
        except InvalidGitRepositoryError as exc:
            raise ValueError(f"指定路径不是 Git 仓库: {path}") from exc

        recent_commits = tuple(self._collect_recent_commits(repo))
        working_tree_diff = self._collect_working_tree_diff(repo)
        branch_name = self._resolve_branch_name(repo)

        return GitRepoParseResult(
            repo_path=str(Path(repo.working_tree_dir or path).resolve()),
            branch_name=branch_name,
            recent_commits=recent_commits,
            working_tree_diff=working_tree_diff,
        )

    def _collect_recent_commits(self, repo: Repo) -> list[CommitInfo]:
        """提取最近 10 次提交。"""

        commits: list[CommitInfo] = []
        if not self._has_head_commit(repo):
            return commits

        for commit in repo.iter_commits(max_count=10):
            commits.append(
                CommitInfo(
                    hexsha=commit.hexsha,
                    short_sha=commit.hexsha[:7],
                    author_name=commit.author.name,
                    authored_at=commit.authored_datetime.isoformat(),
                    summary=commit.summary,
                )
            )
        return commits

    def _collect_working_tree_diff(self, repo: Repo) -> str:
        """提取当前工作区相对 HEAD 的 diff。"""

        if not self._has_head_commit(repo):
            return ""

        diff_text = repo.git.diff("HEAD")
        untracked_files = repo.untracked_files
        if not untracked_files:
            return diff_text

        untracked_summary = "\n".join(f"UNTRACKED: {file_path}" for file_path in untracked_files)
        if not diff_text:
            return untracked_summary
        return f"{diff_text}\n\n{untracked_summary}"

    def _resolve_branch_name(self, repo: Repo) -> str:
        """获取当前分支名称。"""

        if repo.head.is_detached:
            return "DETACHED"
        return repo.active_branch.name

    def _has_head_commit(self, repo: Repo) -> bool:
        """判断仓库是否已有首个提交。"""

        try:
            _ = repo.head.commit
            return True
        except ValueError:
            return False
