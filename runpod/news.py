from __future__ import annotations

import os
import sys


def search_preparation_news(query: str) -> list[dict[str, str]]:
    if not query.strip():
        return []

    from retrieval import load_feature_chat_rag

    load_feature_chat_rag()
    from rag.naver_news import search_naver_news

    try:
        items = search_naver_news(
            query,
            display=int(os.getenv("PREPARATION_NEWS_DISPLAY", "5")),
            sort=os.getenv("PREPARATION_NEWS_SORT", "sim"),
            timeout=float(os.getenv("PREPARATION_NEWS_TIMEOUT", "10")),
        )
    except Exception as exc:
        print(f"[WARN] Naver news search failed for preparation: {exc}", file=sys.stderr)
        return []

    docs: list[dict[str, str]] = []
    for item in items:
        url = item.originallink or item.link
        docs.append(
            {
                "category": "external_news",
                "source": url,
                "title": item.title,
                "text": item.description,
                "pub_date": item.pub_date,
                "url": url,
            }
        )
    return docs
