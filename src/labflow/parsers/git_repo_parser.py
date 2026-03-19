"""Git 仓库解析器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

CODE_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".py",)
TEXT_FILE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".sh",
    ".ps1",
    ".html",
    ".css",
    ".scss",
    ".vue",
    ".sql",
)


@dataclass(frozen=True)
class CommitInfo:
    """提交摘要。"""

    hexsha: str
    short_sha: str
    author_name: str
    authored_at: str
    summary: str


@dataclass(frozen=True)
class SourceFile:
    """可供右侧代码检查器直接展示的源码文件。"""

    relative_path: str
    content: str
    language: str = "python"


@dataclass(frozen=True)
class GitRepoParseResult:
    """Git 仓库解析结果。"""

    repo_path: str
    branch_name: str
    recent_commits: tuple[CommitInfo, ...]
    working_tree_diff: str
    source_files: tuple[SourceFile, ...] = ()
    source_type: str = "git"

    @property
    def is_git_repo(self) -> bool:
        """区分真实 Git 仓库和普通源码目录。"""

        return self.source_type == "git"


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
        except InvalidGitRepositoryError:
            return self._parse_unversioned_directory(path)

        recent_commits = tuple(self._collect_recent_commits(repo))
        working_tree_diff = self._collect_working_tree_diff(repo)
        branch_name = self._resolve_branch_name(repo)
        repo_root = Path(repo.working_tree_dir or path).resolve()

        return GitRepoParseResult(
            repo_path=str(repo_root),
            branch_name=branch_name,
            recent_commits=recent_commits,
            working_tree_diff=working_tree_diff,
            source_files=self._collect_source_files(repo_root, CODE_FILE_EXTENSIONS),
            source_type="git",
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

    def _parse_unversioned_directory(self, path: Path) -> GitRepoParseResult:
        """普通源码目录也要能继续分析，我把它视为“待纳管工作区”。"""

        working_tree_diff = self._build_directory_snapshot_diff(path)
        return GitRepoParseResult(
            repo_path=str(path.resolve()),
            branch_name="UNVERSIONED",
            recent_commits=(),
            working_tree_diff=working_tree_diff,
            source_files=self._collect_source_files(path.resolve(), CODE_FILE_EXTENSIONS),
            source_type="directory",
        )

    def _build_directory_snapshot_diff(self, path: Path, max_files: int = 20) -> str:
        """把普通目录按新文件快照包装成 diff，复用后续证据构建链路。"""

        diff_chunks: list[str] = []
        for file_path in self._iter_candidate_files(path, max_files=max_files):
            relative_path = file_path.relative_to(path).as_posix()
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            line_count = len(lines)
            preview_lines = lines[:120]
            diff_lines = [
                f"diff --git a/{relative_path} b/{relative_path}",
                "new file mode 100644",
                "--- /dev/null",
                f"+++ b/{relative_path}",
                f"@@ -0,0 +1,{max(line_count, 1)} @@",
            ]
            diff_lines.extend(f"+{line}" for line in preview_lines)
            if line_count > len(preview_lines):
                diff_lines.append("+... (文件较长，已截断预览)")
            diff_chunks.append("\n".join(diff_lines))

        if diff_chunks:
            return "\n\n".join(diff_chunks)
        return (
            "diff --git a/WORKSPACE_SUMMARY b/WORKSPACE_SUMMARY\n"
            "+当前目录下没有可解析的文本源码文件。"
        )

    def _iter_candidate_files(self, path: Path, *, max_files: int) -> list[Path]:
        """限制文件规模，避免 Zip 解压目录一次性把界面拖死。"""

        candidate_files: list[Path] = []
        for file_path in sorted(path.rglob("*")):
            if len(candidate_files) >= max_files:
                break
            if not file_path.is_file():
                continue
            if any(part.startswith(".") for part in file_path.relative_to(path).parts):
                continue
            if file_path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
                continue
            if file_path.stat().st_size > 200_000:
                continue
            candidate_files.append(file_path)
        return candidate_files

    def _collect_source_files(
        self,
        path: Path,
        extensions: tuple[str, ...],
        *,
        max_files: int = 60,
    ) -> tuple[SourceFile, ...]:
        """把源码文件收口成统一结构，给局部联动检索直接复用。"""

        source_files: list[SourceFile] = []
        for file_path in self._iter_code_files(path, extensions=extensions, max_files=max_files):
            relative_path = file_path.relative_to(path).as_posix()
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            source_files.append(
                SourceFile(
                    relative_path=relative_path,
                    content=content,
                    language=self._infer_language(file_path),
                )
            )
        return tuple(source_files)

    def _iter_code_files(
        self,
        path: Path,
        *,
        extensions: tuple[str, ...],
        max_files: int,
    ) -> list[Path]:
        """扫描代码文件时只保留真正适合做右栏阅读的文本源码。"""

        code_files: list[Path] = []
        for file_path in sorted(path.rglob("*")):
            if len(code_files) >= max_files:
                break
            if not file_path.is_file():
                continue
            if any(part.startswith(".") for part in file_path.relative_to(path).parts):
                continue
            if file_path.suffix.lower() not in extensions:
                continue
            if file_path.stat().st_size > 200_000:
                continue
            code_files.append(file_path)
        return code_files

    def _infer_language(self, file_path: Path) -> str:
        """给代码块标注语言，保证右栏高亮稳定。"""

        if file_path.suffix.lower() == ".py":
            return "python"
        return "text"
