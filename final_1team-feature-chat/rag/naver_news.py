from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class NewsItem:
    title: str
    description: str
    originallink: str
    link: str
    pub_date: str


def _clean_html(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub("", text)
    return " ".join(text.split())


def search_naver_news(
    query: str,
    display: int = 5,
    sort: str = "sim",
    timeout: float = 10.0,
) -> list[NewsItem]:
    client_id = os.getenv("NAVER_CLIENT_ID") or os.getenv("NAVER_NEWS_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET") or os.getenv("NAVER_NEWS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Missing NAVER_CLIENT_ID/NAVER_CLIENT_SECRET in .env.")

    params = urllib.parse.urlencode(
        {
            "query": query,
            "display": max(1, min(int(display), 100)),
            "start": 1,
            "sort": sort if sort in {"sim", "date"} else "sim",
        }
    )
    request = urllib.request.Request(
        f"{NAVER_NEWS_ENDPOINT}?{params}",
        headers={
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    items: list[NewsItem] = []
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        items.append(
            NewsItem(
                title=_clean_html(item.get("title")),
                description=_clean_html(item.get("description")),
                originallink=str(item.get("originallink") or ""),
                link=str(item.get("link") or ""),
                pub_date=str(item.get("pubDate") or ""),
            )
        )
    return items


def news_context(items: list[NewsItem], max_chars: int = 3000) -> str:
    parts: list[str] = []
    used = 0
    for idx, item in enumerate(items, 1):
        url = item.originallink or item.link
        block = (
            f"[N{idx}] 뉴스: {item.title}\n"
            f"날짜: {item.pub_date}\n"
            f"요약: {item.description}\n"
            f"URL: {url}"
        )
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)].rstrip()
        if block:
            parts.append(block)
            used += len(block) + 2
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def news_source_references(items: list[NewsItem]) -> list[str]:
    refs: list[str] = []
    for idx, item in enumerate(items, 1):
        url = item.originallink or item.link
        summary = item.description or item.pub_date
        refs.append(f"[{idx}] {item.title} {summary} | {url}")
    return refs
