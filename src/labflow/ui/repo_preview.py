"""首页代码目录预览相关的纯函数。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class RepoPreviewGroup:
    """首页里单个目录分组的预览信息。"""

    root_name: str
    file_count: int
    children: tuple[str, ...]
    hidden_child_count: int = 0


@dataclass(frozen=True)
class LandingRepoPreview:
    """首页代码目录预览卡片需要的展示数据。"""

    source_label: str
    detail_label: str
    groups: tuple[RepoPreviewGroup, ...]


def build_landing_repo_preview(
    *,
    relative_paths: tuple[str, ...],
    source_type: str,
    branch_name: str,
    max_groups: int = 6,
    max_children: int = 4,
) -> LandingRepoPreview | None:
    """把源码路径整理成首页可读的轻量目录预览。"""

    unique_paths = tuple(dict.fromkeys(path for path in relative_paths if path))
    if not unique_paths:
        return None

    grouped_children: dict[str, list[str]] = {}
    grouped_files: dict[str, int] = {}

    for relative_path in unique_paths:
        parts = tuple(part for part in relative_path.split("/") if part)
        if not parts:
            continue

        if len(parts) == 1:
            root_name = "根目录"
            child_label = parts[0]
        else:
            root_name = parts[0]
            child_name = parts[1]
            child_label = f"{child_name}/" if len(parts) > 2 else child_name

        grouped_children.setdefault(root_name, [])
        grouped_files[root_name] = grouped_files.get(root_name, 0) + 1
        if child_label not in grouped_children[root_name]:
            grouped_children[root_name].append(child_label)

    preview_groups = tuple(
        _build_preview_group(
            root_name=root_name,
            children=tuple(children),
            file_count=grouped_files.get(root_name, 0),
            max_children=max_children,
        )
        for root_name, children in sorted(
            grouped_children.items(),
            key=lambda item: (-grouped_files.get(item[0], 0), item[0]),
        )[:max_groups]
    )

    source_label = "Git 仓库预览" if source_type == "git" else "代码目录预览"
    if source_type == "git" and branch_name and branch_name != "DETACHED":
        source_label = f"{source_label} · {branch_name}"

    total_files = len(unique_paths)
    detail_label = f"基于前 {total_files} 个源码文件整理出的目录结构速览"
    return LandingRepoPreview(
        source_label=source_label,
        detail_label=detail_label,
        groups=preview_groups,
    )


def _build_preview_group(
    *,
    root_name: str,
    children: tuple[str, ...],
    file_count: int,
    max_children: int,
) -> RepoPreviewGroup:
    visible_children = children[:max_children]
    hidden_child_count = max(0, len(children) - len(visible_children))
    return RepoPreviewGroup(
        root_name=root_name,
        file_count=file_count,
        children=visible_children,
        hidden_child_count=hidden_child_count,
    )


def build_repo_preview_html(preview: LandingRepoPreview) -> str:
    """把目录预览数据渲染成首页卡片 HTML。"""

    group_html = "".join(
        (
            '<div class="repo-preview-group">'
            '<div class="repo-preview-group-head">'
            f'<span class="repo-preview-root">{escape(group.root_name)}</span>'
            f'<span class="repo-preview-count">{group.file_count} 文件</span>'
            "</div>"
            '<div class="repo-preview-children">'
            f"{''.join(_build_child_chip_html(child) for child in group.children)}"
            + (
                f'<span class="repo-preview-child repo-preview-child-muted">'
                f"+{group.hidden_child_count} 项</span>"
                if group.hidden_child_count
                else ""
            )
            + "</div>"
            "</div>"
        )
        for group in preview.groups
    )
    return (
        '<div class="repo-preview-shell">'
        '<div class="repo-preview-head">'
        f'<div class="repo-preview-title">{escape(preview.source_label)}</div>'
        f'<div class="repo-preview-desc">{escape(preview.detail_label)}</div>'
        "</div>"
        f"{group_html}"
        "</div>"
    )


def _build_child_chip_html(child_name: str) -> str:
    return f'<span class="repo-preview-child">{escape(child_name)}</span>'
