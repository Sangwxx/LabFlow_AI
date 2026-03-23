"""Semantic Scholar 论文元数据补充客户端。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SemanticScholarPaper:
    """从 Semantic Scholar 返回的精简论文元数据。"""

    title: str
    authors: tuple[str, ...]
    abstract: str
    year: int | None
    citation_count: int | None
    venue: str | None
    url: str | None


class SemanticScholarClient:
    """首页论文信息卡只做轻量查询，不把异常传播到主流程。"""

    SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
    SEARCH_FIELDS = "title,authors,abstract,year,citationCount,venue,url"

    def __init__(self, *, timeout: float = 4.0) -> None:
        self._timeout = timeout

    def search_by_title(self, title: str) -> SemanticScholarPaper | None:
        normalized_title = " ".join(title.split())
        if len(normalized_title) < 16:
            return None

        params = urlencode(
            {
                "query": normalized_title,
                "limit": 1,
                "fields": self.SEARCH_FIELDS,
            }
        )
        request = Request(
            f"{self.SEARCH_ENDPOINT}?{params}",
            headers={"User-Agent": "LabFlow/1.0"},
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        papers = payload.get("data")
        if not isinstance(papers, list) or not papers:
            return None

        paper = papers[0]
        if not isinstance(paper, dict):
            return None

        return SemanticScholarPaper(
            title=str(paper.get("title", "")).strip() or normalized_title,
            authors=tuple(
                str(author.get("name", "")).strip()
                for author in paper.get("authors", [])
                if isinstance(author, dict) and str(author.get("name", "")).strip()
            ),
            abstract=str(paper.get("abstract", "")).strip(),
            year=_coerce_int(paper.get("year")),
            citation_count=_coerce_int(paper.get("citationCount")),
            venue=str(paper.get("venue", "")).strip() or None,
            url=str(paper.get("url", "")).strip() or None,
        )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
