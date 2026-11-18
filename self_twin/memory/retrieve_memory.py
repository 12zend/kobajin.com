#!/usr/bin/env python3
"""Privacy-conservative local retrieval for confirmed self memories."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "memory_database.sqlite"
MEMORY_INTENT = re.compile(
    r"(?:前に|以前|前回|この前|昔|過去|去年|覚えて|覚え|いつ|誰|何を作|どのプロジェクト|話してた|言ってた|使ってた)"
)
WORD_RE = re.compile(r"[A-Za-z0-9_+#.-]{2,}")
COMMON_JA_BIGRAMS = {
    "前に", "何だ", "だっけ", "って", "した", "して", "いる", "いた", "てい",
    "から", "ので", "のは", "のが", "こと", "もの", "です", "ます", "言っ",
    "本人", "当時", "現在", "過去", "一つ", "など", "って言", "うか", "に好",
    "好き", "きな", "たっ", "っけ", "てた", "て言", "何て",
}


def should_retrieve(query: str) -> bool:
    """Only retrieve when the user asks about prior events or personal facts."""
    return bool(MEMORY_INTENT.search(query))


def query_terms(query: str) -> set[str]:
    terms = {
        match.group(0).casefold()
        for match in WORD_RE.finditer(query)
        if match.group(0).casefold() not in {"前回", "今回", "何か", "どれ"}
    }
    cjk = "".join(ch for ch in query if "\u3040" <= ch <= "\u9fff")
    terms.update(
        cjk[i : i + 2]
        for i in range(max(0, len(cjk) - 1))
        if cjk[i : i + 2] not in COMMON_JA_BIGRAMS
    )
    if len(cjk) == 1:
        terms.add(cjk)
    return terms


def retrieve(query: str, limit: int = 5) -> list[dict[str, object]]:
    if not should_retrieve(query) or not DB_PATH.exists():
        return []
    terms = query_terms(query)
    if not terms:
        return []
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """SELECT memory_id, content, category, observed_at, source_message_id,
                      confidence, is_current, could_be_stale
               FROM memories
               WHERE subject = 'self' AND review_status = 'confirmed' AND confidence >= 0.8"""
        ).fetchall()
    # Named Latin project/product terms are strong anchors; don't let generic
    # Japanese bigrams pull in unrelated memories when one is present.
    latin_terms = {term for term in terms if re.fullmatch(r"[a-z0-9_+#.-]+", term)}
    search_terms = latin_terms or terms
    scored = []
    for row in rows:
        content = f"{row[1]} {row[2]}".casefold()
        score = sum(term in content for term in search_terms)
        minimum_score = 1 if latin_terms else 2
        if score >= minimum_score:
            scored.append((score, row))
    scored.sort(key=lambda item: (item[0], item[1][5]), reverse=True)
    return [
        {
            "memory_id": row[0], "content": row[1], "category": row[2],
            "observed_at": row[3], "source_message_id": row[4],
            "confidence": row[5], "is_current": bool(row[6]),
            "could_be_stale": bool(row[7]),
        }
        for _, row in scored[:limit]
    ]
