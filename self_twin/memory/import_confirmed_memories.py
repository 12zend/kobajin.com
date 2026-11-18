#!/usr/bin/env python3
"""Import the small set of source-cited, explicit self memories into SQLite."""

import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSONL = HERE / "personal_memory.jsonl"
DB = HERE / "memory_database.sqlite"


def main() -> None:
    records = [json.loads(line) for line in JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    with sqlite3.connect(DB) as connection:
        connection.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))
        for record in records:
            if record.get("subject") != "self" or record.get("review_status") != "confirmed":
                raise ValueError("Only explicitly confirmed self memories can be imported")
            connection.execute(
                """INSERT INTO memories
                   (memory_id, content, category, observed_at, source_message_id, confidence,
                    is_current, could_be_stale, subject, review_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'self', 'confirmed')
                   ON CONFLICT(memory_id) DO UPDATE SET
                     content=excluded.content, category=excluded.category,
                     observed_at=excluded.observed_at, source_message_id=excluded.source_message_id,
                     confidence=excluded.confidence, is_current=excluded.is_current,
                     could_be_stale=excluded.could_be_stale, review_status='confirmed'""",
                (
                    record["memory_id"], record["content"], record["category"],
                    record["observed_at"], record["source_message_id"], record["confidence"],
                    int(record["is_current"]), int(record["could_be_stale"]),
                ),
            )
        count = connection.execute(
            "SELECT count(*) FROM memories WHERE subject='self' AND review_status='confirmed'"
        ).fetchone()[0]
    print(f"Imported {len(records)} source-cited self memories; confirmed total={count}")


if __name__ == "__main__":
    main()
