#!/usr/bin/env python3
"""Audit and prepare a privacy-filtered, chronological SFT dataset.

The script uses only the Python standard library. It never uploads source data.
Raw author identifiers are used only to establish the three self accounts and
anonymous speaker labels; identifiers are not emitted in the training JSONL.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DISCORD_EPOCH_MS = 1420070400000
SELF_AUTHOR_IDS = {
    "1120529566138634275": "zend3d",
    "1206039595129639002": "kobajin_",
    "1262554507233984576": "12zend",
}
EXPECTED_COUNTS = {
    "1120529566138634275": 26806,
    "1206039595129639002": 20313,
    "1262554507233984576": 5567,
}
DISPLAY = {
    "1120529566138634275": "12zend",
    "1206039595129639002": "ルイボスティー",
    "1262554507233984576": "triangle.area",
}
JST = dt.timezone(dt.timedelta(hours=9), name="JST")
TIMESTAMP_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(\d{1,2}):(\d{2})")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?81[- ]?|0)\d{1,4}[- ]?\d{2,4}[- ]?\d{3,4}(?!\d)")
DISCORD_MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
SNOWFLAKE_TEXT_RE = re.compile(r"(?<!\d)\d{17,20}(?!\d)")
SECRET_ASSIGN_RE = re.compile(
    r"(?i)(?:password|passwd|api[_ -]?key|access[_ -]?token|secret|パスワード|認証キー)\s*[:=：]\s*\S+"
)
CONTACT_CONTEXT_RE = re.compile(
    r"(?i)(?:電話番号|携帯番号|連絡先|住所|自宅|メールアドレス|phone|mobile|address|contact)"
)
COMMAND_RE = re.compile(r"^\s*(?:/|!|\.)\S")
URL_ONLY_RE = re.compile(r"^(?:https?://\S+\s*)+$", re.IGNORECASE)
QUOTE_LINE_RE = re.compile(r"(?m)^\s*>")
REACTION_RE = re.compile(
    r"(?i)^(?:w{1,}|ｗ{1,}|草+|笑+|www+|lol+|え+|あ+|うん|はい|そう|なるほど|[!?！？。、…]+|[\U0001F300-\U0001FAFF\u2600-\u27BF]+)$"
)
TOPIC_SHIFT_RE = re.compile(
    r"(?:話(?:題)?(?:変わる|変える)(?:けど|けれど|けどさ|けれども|んだけど)?|別の話(?:だけど|だけどさ|に)?|ところで|そういえば)"
)

SYSTEM_PROMPT = "次のグループ会話の流れに合わせて、自然に返事してください。"


def snowflake_datetime(message_id: Any) -> dt.datetime | None:
    try:
        n = int(str(message_id))
        if n <= 0 or n >= 2**64:
            return None
        milliseconds = (n >> 22) + DISCORD_EPOCH_MS
        return dt.datetime.fromtimestamp(milliseconds / 1000, tz=dt.timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def displayed_datetime(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    match = TIMESTAMP_RE.search(value)
    if not match:
        return None
    try:
        return dt.datetime(*map(int, match.groups()), tzinfo=JST)
    except ValueError:
        return None


def message_time(message: dict[str, Any]) -> dt.datetime | None:
    # Message IDs were checked against displayed timestamps during the audit.
    return snowflake_datetime(message.get("message_id")) or displayed_datetime(
        message.get("timestamp")
    )


def author_id(message: dict[str, Any]) -> str:
    author = message.get("author") or {}
    return str(author.get("user_id", ""))


def reply_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    ref = message.get("reply_to")
    return ref if isinstance(ref, dict) else None


def reply_id(message: dict[str, Any]) -> str | None:
    ref = reply_payload(message)
    if not ref:
        return None
    value = ref.get("message_id", ref.get("id"))
    return str(value) if value is not None else None


def normalize_line(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def privacy_reason(text: str) -> str | None:
    if SECRET_ASSIGN_RE.search(text):
        return "possible_credential"
    if EMAIL_RE.search(text):
        return "email_address"
    if PHONE_RE.search(text):
        return "phone_number"
    if CONTACT_CONTEXT_RE.search(text) and re.search(r"\d{2,}|@", text):
        return "contact_or_address_detail"
    return None


def build_name_replacements(messages: list[dict[str, Any]]) -> list[tuple[re.Pattern[str], str]]:
    names: set[str] = set()
    for message in messages:
        authors = [message.get("author") or {}]
        reply = reply_payload(message)
        if reply:
            authors.append(reply.get("author") or {})
        for author in authors:
            uid = str(author.get("user_id", ""))
            for key in ("username", "display_name"):
                value = author.get(key)
                if isinstance(value, str) and len(value.strip()) >= 2:
                    names.add(value.strip())
            if uid in SELF_AUTHOR_IDS:
                names.update((SELF_AUTHOR_IDS[uid], DISPLAY[uid]))
    replacements: list[tuple[re.Pattern[str], str]] = []
    for name in sorted(names, key=len, reverse=True):
        if not name:
            continue
        if re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])", re.I)
        else:
            pattern = re.compile(re.escape(name), re.I)
        replacements.append((pattern, "【参加者】"))
    return replacements


def anonymize(text: str, replacements: list[tuple[re.Pattern[str], str]]) -> str:
    value = URL_RE.sub("【URL】", text)
    value = EMAIL_RE.sub("【連絡先】", value)
    value = PHONE_RE.sub("【連絡先】", value)
    value = DISCORD_MENTION_RE.sub("【参加者】", value)
    value = re.sub(r"(?<!\w)@[A-Za-z0-9_.-]{2,}", "【参加者】", value)
    value = SNOWFLAKE_TEXT_RE.sub("【ID】", value)
    for pattern, replacement in replacements:
        value = pattern.sub(replacement, value)
    return value.strip()


def reason_for_target(text: str) -> str | None:
    if not text.strip():
        return "blank"
    reason = privacy_reason(text)
    if reason:
        return reason
    if URL_ONLY_RE.fullmatch(text.strip()):
        return "url_only"
    if COMMAND_RE.match(text):
        return "system_or_bot_command"
    lines = text.splitlines() or [text]
    quoted = sum(len(line) for line in lines if line.lstrip().startswith(">"))
    if len(text) >= 100 and quoted / max(1, len(text)) >= 0.6:
        return "copied_quote"
    return None


def category_for(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= 8 or REACTION_RE.fullmatch(stripped):
        return "short_reaction"
    if re.search(r"[?？]", text):
        return "question_or_prompt"
    if len(stripped) >= 120:
        return "long_explanation_or_story"
    if re.search(r"(?i)(?:https?://|\b(?:python|blender|after effects|discord|qwen|ai|api|GPU|LoRA)\b|コード|プログラム|モデル|レンダ|動画編集|映像)", text):
        return "technical_or_project"
    if re.search(r"(?:w{2,}|ｗ{2,}|草|笑|www|lol|!?！{1,}|!?？{1,})", text, re.I):
        return "reaction_or_joke"
    return "conversation"


def context_speaker(message: dict[str, Any], self_ids: set[str], label_map: dict[str, str]) -> str:
    uid = author_id(message)
    if uid in self_ids:
        return "自分"
    if uid not in label_map:
        label_map[uid] = f"参加者{chr(ord('A') + len(label_map))}"
    return label_map[uid]


def safe_context_text(message: dict[str, Any], replacements: list[tuple[re.Pattern[str], str]]) -> str:
    raw = normalize_line(message.get("content"))
    if not raw.strip() or privacy_reason(raw):
        return ""
    if COMMAND_RE.match(raw) or URL_ONLY_RE.fullmatch(raw.strip()):
        return ""
    return anonymize(raw, replacements)


def exact_duplicate_ids(messages: list[dict[str, Any]], self_ids: set[str]) -> set[str]:
    # Drop rapid duplicate long posts, while keeping normal repeated short reactions.
    seen: dict[str, tuple[float, int]] = {}
    excluded: set[str] = set()
    for message in messages:
        if author_id(message) not in self_ids:
            continue
        raw = normalize_line(message.get("content")).strip()
        if len(raw) < 24:
            continue
        key = re.sub(r"\s+", " ", raw).casefold()
        stamp = message_time(message)
        if stamp is None:
            continue
        seconds = stamp.timestamp()
        previous = seen.get(key)
        if previous and seconds - previous[0] <= 600:
            excluded.add(str(message.get("message_id")))
            seen[key] = (seconds, previous[1] + 1)
        else:
            seen[key] = (seconds, 1)
    return excluded


def collect_sessions(messages: list[dict[str, Any]], gap_seconds: int) -> list[list[int]]:
    sessions: list[list[int]] = []
    current: list[int] = []
    last_time: dt.datetime | None = None
    for index, message in enumerate(messages):
        stamp = message_time(message)
        if current and stamp and last_time and (stamp - last_time).total_seconds() > gap_seconds:
            sessions.append(current)
            current = []
        current.append(index)
        if stamp:
            last_time = stamp
    if current:
        sessions.append(current)
    return sessions


def collect_bursts(
    messages: list[dict[str, Any]], sessions: list[list[int]], self_ids: set[str], gap_seconds: int
) -> tuple[list[dict[str, Any]], int]:
    bursts: list[dict[str, Any]] = []
    topic_shift_boundaries = 0
    for session_number, indices in enumerate(sessions):
        current: list[int] = []
        last_time: dt.datetime | None = None
        for index in indices:
            message = messages[index]
            if author_id(message) not in self_ids:
                if current:
                    bursts.append({"session_number": session_number, "indices": current})
                    current = []
                last_time = None
                continue
            stamp = message_time(message)
            if current and TOPIC_SHIFT_RE.search(normalize_line(message.get("content"))):
                bursts.append({"session_number": session_number, "indices": current})
                current = []
                topic_shift_boundaries += 1
            if current and stamp and last_time and (stamp - last_time).total_seconds() > gap_seconds:
                bursts.append({"session_number": session_number, "indices": current})
                current = []
            current.append(index)
            last_time = stamp
        if current:
            bursts.append({"session_number": session_number, "indices": current})
    return bursts, topic_shift_boundaries


def split_sessions(sample_counts: list[int]) -> tuple[list[str], dict[str, int]]:
    total = sum(sample_counts)
    if total == 0:
        return [], {"train": 0, "validation": 0, "test": 0}
    cumulative = [0]
    for count in sample_counts:
        cumulative.append(cumulative[-1] + count)
    n = len(sample_counts)
    first = min(range(1, n + 1), key=lambda i: abs(cumulative[i] - total * 0.8))
    if first >= n:
        first = max(1, n - 2)
    second_candidates = range(first + 1, n + 1) if first < n else range(first, n + 1)
    second = min(second_candidates, key=lambda i: abs(cumulative[i] - total * 0.9))
    if second <= first:
        second = min(n, first + 1)
    labels = ["train" if i < first else "validation" if i < second else "test" for i in range(n)]
    counts = {split: sum(count for count, label in zip(sample_counts, labels) if label == split)
              for split in ("train", "validation", "test")}
    return labels, counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def try_content_token_counter(tokenizer_path: Path | None):
    if not tokenizer_path or not tokenizer_path.exists():
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return lambda text: len(tokenizer.encode(text).ids)
    except Exception:
        return None


def try_chat_sample_token_counter(model_id: str | None, revision: str | None, cache_dir: Path):
    if not model_id or not revision:
        return None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )

        def count(messages: list[dict[str, str]]) -> int:
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            ids = encoded["input_ids"] if hasattr(encoded, "keys") and "input_ids" in encoded else encoded
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return len(ids)

        return count
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/Users/kobajin/Downloads/chat_messages.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--session-gap-hours", type=float, default=6)
    parser.add_argument("--burst-gap-minutes", type=float, default=10)
    parser.add_argument("--context-messages", type=int, default=12)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / "hf_home" / "hub")
    args = parser.parse_args()

    source_path = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise SystemExit("Input JSON must contain a messages array")
    messages = [m for m in raw_messages if isinstance(m, dict)]

    all_ids = {str(m.get("message_id")) for m in messages}
    self_ids = set(SELF_AUTHOR_IDS)
    account_counts = collections.Counter(author_id(m) for m in messages if author_id(m) in self_ids)
    if dict(account_counts) != EXPECTED_COUNTS:
        raise SystemExit(f"Self account counts differ from expected: {dict(account_counts)}")
    self_adjacent_pairs = sum(
        1
        for previous, current in zip(messages, messages[1:])
        if author_id(previous) in self_ids and author_id(current) in self_ids
    )
    raw_contiguous_runs = 0
    raw_multi_post_runs = 0
    in_self_run = False
    run_length = 0
    for message in messages:
        if author_id(message) in self_ids:
            if not in_self_run:
                raw_contiguous_runs += 1
                in_self_run = True
                run_length = 0
            run_length += 1
        elif in_self_run:
            raw_multi_post_runs += run_length > 1
            in_self_run = False
    if in_self_run:
        raw_multi_post_runs += run_length > 1

    # Verify Discord snowflake time against all export timestamps that can be parsed.
    comparable = 0
    matching = 0
    max_delta_seconds = 0.0
    monotonic = True
    previous_id = 0
    for message in messages:
        try:
            numeric_id = int(str(message.get("message_id")))
        except (TypeError, ValueError):
            continue
        if numeric_id < previous_id:
            monotonic = False
        previous_id = numeric_id
        snowflake = snowflake_datetime(numeric_id)
        displayed = displayed_datetime(message.get("timestamp"))
        if snowflake and displayed:
            comparable += 1
            delta = abs((snowflake.astimezone(JST) - displayed).total_seconds())
            max_delta_seconds = max(max_delta_seconds, delta)
            if delta <= 90:
                matching += 1
    if not monotonic or comparable == 0 or matching / comparable < 0.995:
        raise SystemExit("Snowflake chronology did not validate; refusing to split on IDs")

    replacements = build_name_replacements(messages)
    duplicate_ids = exact_duplicate_ids(messages, self_ids)
    session_gap = int(args.session_gap_hours * 3600)
    burst_gap = int(args.burst_gap_minutes * 60)
    sessions = collect_sessions(messages, session_gap)
    bursts, topic_shift_boundaries = collect_bursts(messages, sessions, self_ids, burst_gap)
    excluded: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    candidate_counts = [0] * len(sessions)
    categories = collections.Counter()
    category_chars = collections.Counter()
    source_self_messages = 0
    included_source_posts = 0
    excluded_reason_counts = collections.Counter()
    review_flags = collections.Counter()
    all_reply_messages = 0
    self_reply_messages = 0
    resolved_reply_count = 0

    for m in messages:
        if m.get("reply_to"):
            all_reply_messages += 1
            if author_id(m) in self_ids:
                self_reply_messages += 1
            ref_id = reply_id(m)
            if ref_id and ref_id in all_ids:
                resolved_reply_count += 1

    message_index = {str(m.get("message_id")): i for i, m in enumerate(messages)}
    session_labels = ["" for _ in sessions]

    for burst in bursts:
        indices = burst["indices"]
        session_number = burst["session_number"]
        source_self_messages += len(indices)
        cleaned_parts: list[tuple[int, str]] = []
        burst_reasons: dict[int, str] = {}
        for index in indices:
            message = messages[index]
            mid = str(message.get("message_id"))
            raw = normalize_line(message.get("content"))
            reason = "rapid_duplicate" if mid in duplicate_ids else reason_for_target(raw)
            if reason:
                burst_reasons[index] = reason
                continue
            cleaned = anonymize(raw, replacements)
            if cleaned:
                cleaned_parts.append((index, cleaned))
            else:
                burst_reasons[index] = "blank_after_redaction"

        for index, reason in burst_reasons.items():
            excluded.append({"message_id": str(messages[index].get("message_id")), "reason": reason})
            excluded_reason_counts[reason] += 1
        if not cleaned_parts:
            continue

        target = "\n".join(text for _, text in cleaned_parts)
        first_index = indices[0]
        first_message = messages[first_index]
        session_indices = sessions[session_number]
        position = session_indices.index(first_index)
        prior_indices = session_indices[:position]
        recent = prior_indices[-args.context_messages :]

        # A reply target is inserted first when it is in this split-safe session.
        reply_index = None
        reply_fallback: dict[str, Any] | None = None
        primary_reply_id = None
        reply_cross_session = False
        for index in indices:
            rid = reply_id(messages[index])
            if rid and primary_reply_id is None:
                primary_reply_id = rid
            if rid and rid in message_index:
                candidate_reply_index = message_index[rid]
                if candidate_reply_index in session_indices:
                    reply_index = candidate_reply_index
                else:
                    reply_cross_session = True
                break
            elif rid and reply_payload(messages[index]):
                reply_fallback = reply_payload(messages[index])
        label_map: dict[str, str] = {}
        context_lines: list[str] = []
        if reply_index is not None and reply_index < first_index:
            reply_message = messages[reply_index]
            reply_text = safe_context_text(reply_message, replacements)
            if reply_text:
                stamp = message_time(reply_message)
                time_label = stamp.astimezone(JST).strftime("%m-%d %H:%M") if stamp else "時刻不明"
                speaker = context_speaker(reply_message, self_ids, label_map)
                context_lines.append(f"【返信先 {time_label}】【{speaker}】{reply_text}")
        elif reply_fallback:
            reply_text = safe_context_text(reply_fallback, replacements)
            if reply_text:
                author = reply_fallback.get("author") or {}
                stamp = snowflake_datetime(reply_fallback.get("message_id"))
                time_label = stamp.astimezone(JST).strftime("%m-%d %H:%M") if stamp else "時刻不明"
                speaker = context_speaker({"author": author}, self_ids, label_map)
                context_lines.append(f"【返信先 {time_label}】【{speaker}】{reply_text}")

        context_ids: list[str] = []
        for index in recent:
            if index == reply_index:
                continue
            message = messages[index]
            text = safe_context_text(message, replacements)
            if not text:
                continue
            stamp = message_time(message)
            time_label = stamp.astimezone(JST).strftime("%m-%d %H:%M") if stamp else "時刻不明"
            speaker = context_speaker(message, self_ids, label_map)
            context_lines.append(f"{time_label} 【{speaker}】{text}")
            context_ids.append(str(message.get("message_id")))
        if not context_lines:
            context_lines = ["（この会話セッション内に直前の発言なし）"]
        context = "\n".join(context_lines)
        category = category_for(target)
        categories[category] += 1
        category_chars[category] += len(target)
        source_included = len(cleaned_parts)
        included_source_posts += source_included
        if category == "short_reaction" and not primary_reply_id:
            review_flags["short_reaction_without_explicit_reply"] += 1
        if len(target) >= 700:
            review_flags["long_target_review"] += 1
        first_time = message_time(first_message)
        last_time = message_time(messages[indices[-1]])
        sid = str(messages[session_indices[0]].get("message_id"))
        samples.append(
            {
                "session_number": session_number,
                "session_id": sid,
                "sort_time": first_time.timestamp() if first_time else 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "message_ids": [str(messages[i].get("message_id")) for i, _ in cleaned_parts],
                    "context_message_ids": context_ids,
                    "session_id": sid,
                    "category": category,
                    "target_post_count": source_included,
                    "target_chars": len(target),
                    "low_weight_review": category == "short_reaction" or len(target) >= 700,
                    "reply_message_id": primary_reply_id,
                    "reply_target_resolved": bool(primary_reply_id and primary_reply_id in all_ids),
                    "reply_context_omitted_cross_session": reply_cross_session,
                    "last_target_time": last_time.isoformat() if last_time else None,
                },
            }
        )
        candidate_counts[session_number] += 1

    labels, split_sample_counts = split_sessions(candidate_counts)
    session_labels = labels
    split_samples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for sample in samples:
        label = session_labels[sample["session_number"]]
        sample.pop("session_number")
        sample.pop("sort_time")
        split_samples[label].append(sample)

    for split, rows in split_samples.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output / "excluded_samples.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    content_token_counter = try_content_token_counter(args.tokenizer)
    chat_token_counter = try_chat_sample_token_counter(args.model_id, args.revision, args.cache_dir)
    split_chat_token_counts: dict[str, int | None] = {split: None for split in split_samples}
    split_content_token_counts: dict[str, int | None] = {split: None for split in split_samples}
    sequence_length_distributions: dict[str, dict[str, int | None]] = {}
    if chat_token_counter:
        for split, rows in split_samples.items():
            lengths = [chat_token_counter(row["messages"]) for row in rows]
            ordered = sorted(lengths)
            split_chat_token_counts[split] = sum(lengths)
            sequence_length_distributions[split] = {
                "p50": ordered[len(ordered) // 2] if ordered else None,
                "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))] if ordered else None,
                "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None,
                "max": max(ordered) if ordered else None,
            }
    if content_token_counter:
        for split, rows in split_samples.items():
            split_content_token_counts[split] = sum(
                content_token_counter(message["content"])
                for row in rows
                for message in row["messages"]
            )

    split_session_counts: dict[str, int] = {
        split: sum(1 for label in session_labels if label == split) for split in ("train", "validation", "test")
    }
    split_date_ranges: dict[str, dict[str, str | None]] = {}
    for split in ("train", "validation", "test"):
        split_times = [
            sample["metadata"].get("last_target_time")
            for sample in split_samples[split]
            if sample["metadata"].get("last_target_time")
        ]
        split_date_ranges[split] = {
            "start": min(split_times) if split_times else None,
            "end": max(split_times) if split_times else None,
        }

    manifest = {
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "bytes": source_path.stat().st_size,
            "channel": data.get("channel"),
            "messages": len(messages),
        },
        "self_author_ids": sorted(self_ids),
        "self_account_counts": {
            SELF_AUTHOR_IDS[uid]: {"user_id": uid, "count": account_counts[uid], "expected": EXPECTED_COUNTS[uid]}
            for uid in SELF_AUTHOR_IDS
        },
        "self_total": sum(account_counts.values()),
        "snowflake_validation": {
            "message_ids_monotonic": monotonic,
            "timestamps_compared": comparable,
            "timestamps_within_90_seconds": matching,
            "max_delta_seconds": max_delta_seconds,
            "used_for_order_and_session_gaps": True,
        },
        "reply_to": {
            "all_messages_with_reply": all_reply_messages,
            "self_messages_with_reply": self_reply_messages,
            "references_resolved_to_export_messages": resolved_reply_count,
            "reference_payload_has_author_content_and_message_id": True,
            "cross_session_context_omissions": sum(
                row["metadata"]["reply_context_omitted_cross_session"] for row in samples
            ),
        },
        "preprocessing": {
            "session_gap_hours": args.session_gap_hours,
            "burst_gap_minutes": args.burst_gap_minutes,
            "explicit_topic_shift_boundaries": topic_shift_boundaries,
            "topic_shift_markers": ["話変わる", "話題変わる", "別の話", "ところで", "そういえば"],
            "max_context_messages": args.context_messages,
            "speaker_labels": "anonymous per-session labels; source usernames and IDs omitted",
            "privacy_filters": ["email", "phone", "credential assignments", "contact details with digits", "Discord IDs and mentions"],
            "context_reply_priority": True,
            "burst_posts_joined_with_newlines": True,
            "copied_quote_policy": "exclude target when at least 60% of a >=100-character post is quoted text",
            "tokenizer": {
                "path": str(args.tokenizer) if args.tokenizer else None,
                "model_id": args.model_id,
                "revision": args.revision,
            },
            "token_counts_status": (
                "exact chat-template sequence lengths computed" if chat_token_counter
                else "full chat-template token counts unavailable; content-only counts are separate"
            ),
        },
        "source_self_messages": source_self_messages,
        "adjacent_self_post_pairs": self_adjacent_pairs,
        "raw_uninterrupted_self_runs": raw_contiguous_runs,
        "raw_uninterrupted_multi_post_runs": raw_multi_post_runs,
        "included_target_posts": included_source_posts,
        "excluded_target_posts": len(excluded),
        "excluded_reason_counts": dict(excluded_reason_counts),
        "rapid_duplicate_count": len(duplicate_ids),
        "sessions": len(sessions),
        "self_response_bursts": len(bursts),
        "dataset_samples": len(samples),
        "split_samples": {split: len(rows) for split, rows in split_samples.items()},
        "split_sessions": split_session_counts,
        "split_date_ranges": split_date_ranges,
        "split_token_counts": split_chat_token_counts,
        "split_content_token_counts": split_content_token_counts,
        "sequence_length_distribution": sequence_length_distributions,
        "split_target_characters": {
            split: sum(len(row["messages"][-1]["content"]) for row in rows)
            for split, rows in split_samples.items()
        },
        "category_counts": dict(categories),
        "category_ratios": {
            category: round(count / max(1, len(samples)), 6)
            for category, count in categories.items()
        },
        "category_target_characters": dict(category_chars),
        "review_flags": dict(review_flags),
        "split_policy": "whole 6-hour inactivity sessions; reply context is included when it stays inside the same session, and omitted when it would cross a split; chronological nearest 80/10/10 target-sample boundaries; newest sessions go to test",
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "source_self_counts": {SELF_AUTHOR_IDS[uid]: account_counts[uid] for uid in SELF_AUTHOR_IDS},
        "adjacent_self_pairs": self_adjacent_pairs,
        "raw_uninterrupted_self_runs": raw_contiguous_runs,
        "raw_uninterrupted_multi_post_runs": raw_multi_post_runs,
        "sessions": len(sessions),
        "bursts": len(bursts),
        "samples": len(samples),
        "split_samples": manifest["split_samples"],
        "excluded_posts": len(excluded),
        "excluded_reasons": dict(excluded_reason_counts),
        "categories": dict(categories),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
