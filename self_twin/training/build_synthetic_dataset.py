#!/usr/bin/env python3
"""Build a 2,000-row, evidence-bounded synthetic SFT set.

The case bank contains new, generic situations in topics present in the
Discord corpus. It does not reproduce private messages or make claims about
Kobajin's personal history, preferences, or beliefs. Its purpose is to add
useful, context-sensitive replies that are sparse in ordinary chat logs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "dataset"
CASE_BANK = HERE / "synthetic_case_bank.jsonl"
SYSTEM_PROMPT = "次の会話の流れに合わせて、自然に返事してください。"
ROWS_PER_CASE = 10
VALIDATION_SCENARIOS = 20

INTENTS = (
    "diagnosis",
    "explanation",
    "first_step",
    "concern",
    "verification",
    "alternative",
    "plain_explanation",
    "triage",
    "clarification",
    "brief_summary",
)


def load_cases() -> list[dict[str, str]]:
    with CASE_BANK.open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    required = {
        "topic", "problem", "cause", "why", "step", "check", "concern",
        "tradeoff", "alternative", "analogy", "clarify",
    }
    for index, case in enumerate(cases, start=1):
        missing = required - case.keys()
        if missing:
            raise SystemExit(f"Case {index} is missing fields: {sorted(missing)}")
        if any(not str(case[key]).strip() for key in required):
            raise SystemExit(f"Case {index} contains an empty field")
    if len(cases) != 200:
        raise SystemExit(f"Expected 200 authored cases, found {len(cases)}")
    return cases


def finish(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in "。！？!?」』】":
        text += "。"
    return text


def render_case(case: dict[str, str], case_index: int) -> list[tuple[str, str]]:
    """Render ten different user intents with a hand-authored answer per case."""
    p = case["problem"]
    cause = case["cause"]
    why = case["why"]
    step = case["step"]
    check = case["check"]
    concern = case["concern"]
    analogy = case["analogy"]
    clarify = case["clarify"]
    variants = (case_index * 7) % 8
    diagnoses = (
        f"あー、原因は「{cause}」かも。{why}",
        f"これ、「{cause}」っぽい。{why}",
        f"うーん、「{cause}」もありそう。{why}",
        f"聞いた感じだと「{cause}」かな。{why}",
        f"原因は「{cause}」かもね。{why}",
        f"ここ、「{cause}」が引っかかってそう。{why}",
        f"まず疑うなら「{cause}」。{why}",
        f"あ、まずは「{cause}」かな。{why}",
    )
    explanations = (
        f"「{why}」ってことかな。まず「{clarify}」を見てみると、原因を絞りやすそう。",
        f"たぶん「{why}」。先に{step}。",
        f"ここは「{why}」っぽい。一回「{check}」を確かめるのがよさそう。",
        f"あー、「{why}」。だから「{clarify}」を見たい。",
        f"それなら「{why}」。まず{step}。",
        f"要するに「{why}」ってこと。{check}に出る変化を見ればよさそう。",
        f"これ、「{why}」ってことかな。まずは「{clarify}」を見たい。",
        f"理由は「{why}」。一気に触らず「{check}」を見てみよ。",
    )
    first_steps = (
        f"まず{step}。{why}。そこだけ変えて一回比べるのがよさそう。",
        f"手を付けるなら{step}かな。{why}。他は今いじらなくていいと思う。",
        f"先に{step}。{why}。まずそこだけ見てみよ。",
        f"これなら{step}。ほかを一緒に変えると差が分かりにくいし。",
        f"あー、最初は{step}かな。そこだけ試せば原因を追いやすい。",
        f"一回{step}。そのあと「{check}」を見よ。",
        f"僕なら{step}。まずそこだけで差を見よ。",
        f"まずは{step}。変化があれば次を決めやすい。",
    )
    concerns = (
        f"そこは気になるね。{case['tradeoff']}。まず{step}。その結果を見てから広げるのがよさそう。",
        f"たしかに、「{concern}」は気になるよね。{case['tradeoff']}。まずは「{check}」を見て決めよ。",
        f"そこ心配なの分かる。{case['tradeoff']}。急に全部変えず、{step}。そこで差が見えると思う。",
        f"うん、{case['tradeoff']}。小さく試すなら{step}。気になる影響が出るか見てみよ。",
        f"あー、そこは気になるよね。{case['tradeoff']}。まず「{check}」を見てから範囲を広げたい。",
        f"たしかに？ {case['tradeoff']}。一気に決めず、{step}。変化を見てからでいいと思う。",
        f"その心配は分かる。{case['tradeoff']}。先に{step}。問題なさそうなら広げるのがよさそう。",
        f"そこ悩むね。{case['tradeoff']}。いきなり全部変えず、{step}。影響を見てからでよさそう。",
    )
    checks = (
        f"見るなら「{check}」かな。そこが変わってるかで、直ったか分かりやすい。あわせて「{cause}」が起きる条件も見たい。",
        f"一回「{check}」を見よ。そこが変われば、直ったか判断しやすい。変化がなければ別の原因も考えられる。",
        f"「{check}」を見るのがよさそう。そこに変化が出てるか確認しよ。「{why}」が起きてるかも手掛かりになる。",
        f"判断するなら「{check}」かな。そこが見えれば一旦よさそう。特に「{cause}」が出る条件と比べたい。",
        f"まず「{check}」だけ確認しよ。ここが分かると次へ進めるし、「{why}」かどうかも見えてきそう。",
        f"一回試したあと「{check}」を見る。変わらなかったら「{cause}」以外の所を調べればよさそう。",
        f"目安は「{check}」。そこが安定してれば直ったと言えそう。変化した時の条件も残しておくと、あとで比べやすい。",
        f"そこは「{check}」で見ればよさそう。細かい所はその後かな。「{why}」に近い変化が出るかを見たい。",
    )
    plain_explanations = (
        f"ざっくりいうと、{analogy}。まず{step}。",
        f"難しく言わないなら「{analogy}」みたいなこと。最初は{step}。",
        f"つまり、{analogy}。まず{step}。",
        f"例えると{analogy}って感じ。最初に{step}。",
        f"あー、イメージは{analogy}。なので{step}。",
        f"めっちゃ簡単に言うと、{analogy}。まず{step}。",
        f"要するに{analogy}。そこを確かめるなら{step}かな。",
        f"普通の言い方にすると、{analogy}。まず{step}。",
    )
    triage = (
        f"まず「{check}」だけ見てみて。そこから「{cause}」が関係してるか切り分けられそう。",
        f"最初は「{check}」かな。他の設定はまだ触らなくてよさそう。原因候補の「{cause}」に近い変化があるか見よ。",
        f"一回「{check}」だけ見よ。そこが分かれば次を決めやすいし、「{why}」かどうかも追える。",
        f"うーん、まず「{check}」かな。「{cause}」が起きる条件と比べれば、見る場所をだいぶ絞れそう。",
        f"「{check}」を確認しよ。結果を見てから次の箇所を一つだけ確認すれば、どこで変わったか追いやすい。",
        f"先に見るのは「{check}」。他は今の段階だと後でいいと思う。まず一つだけ結果を取ろ。",
        f"そこはログで「{check}」を見たい。「{cause}」が出ているか分かれば次の確認先を減らせる。",
        f"まず「{check}」を一回だけ確認。そこが分かると話早いし、次に触る場所も決めやすい。",
    )
    clarifications = (
        f"今ほしいのは「{clarify}」かな。そこが分かると「{cause}」の線を見ていいか判断しやすい。",
        f"「{clarify}」が分かると助かる。そこから原因を絞れそうだし、次に「{check}」を確かめるべきかも決められる。",
        f"うーん、先に「{clarify}」を見たい。別の条件もありそうだし、「{check}」の結果も一緒に見て考えたい。",
        f"あー、まず「{clarify}」かな。そこが分かれば次の手を考えられる。「{why}」が当てはまるかも見たい。",
        f"決めつける前に「{clarify}」を見たい。別の条件もありそうだし、「{cause}」だけとは限らない。",
        f"「{clarify}」って今分かる？そこ次第で「{cause}」を疑うか、別の条件から見るか変わりそう。",
        f"まず知りたいのは「{clarify}」。今の情報だけだとまだ決めきれないし、「{check}」がどうなってるかも知りたい。",
        f"うーん、「{clarify}」を確認したい。そこから一個ずつ見ていこ。分かったら次は「{check}」を比べたい。",
    )
    summaries = (
        f"一旦、{step}。{check}を見て次を決めよ。",
        f"先に{step}。結果は「{check}」で確認。",
        f"まずは{step}。見るのは「{check}」。",
        f"{step}。まず「{check}」だけ見よ。",
        f"あー、先に{step}かな。細かい所は後でいいと思う。",
        f"一回{step}。変化が出たかだけ確認しよ。",
        f"今は{step}。見るのは「{check}」かな。",
        f"まず{step}。そこから次を決めよう。",
    )
    return [
        (p, diagnoses[variants]),
        (f"{p} {('なんでこうなるんだろ', 'なんでだろ', 'これってどういう仕組み？', '原因どこだろ', '何が起きてる？')[case_index % 5]}", explanations[variants]),
        (f"{p} {('どこから触ればいい？', 'まず何を試す？', '最初どこを見る？', '一個やるならどれ？', 'どこから確認しよ？')[case_index % 5]}", first_steps[variants]),
        (f"{p} でも{concern}", concerns[variants]),
        (f"{p} {('直ったかどうか、何を見ればいい？', 'どこが変わったら成功？', 'できたって何で分かる？', '確認するポイントある？')[case_index % 4]}", checks[variants]),
        (f"{p} {('他のやり方もある？', '別の方法だと何がいい？', '他の選択肢は？', 'これ以外の方法ある？')[case_index % 4]}", f"{case['alternative']}。決める前に「{check}」を比べると、どっちが合うか見えやすい。"),
        (f"{p} {('初めて聞く人向けにいうと？', 'もっと簡単に説明すると？', 'ざっくり言うとどういうこと？', '一言で例えると？')[case_index % 4]}", plain_explanations[variants]),
        (f"{p} {('まず一個だけ確認するなら？', '最初に見るところは？', '原因を分けるならどこ？', '一つだけログ取るなら？')[case_index % 4]}", triage[variants]),
        (f"{p} {('原因を絞るには何の情報がいる？', '追加で何を見ればいい？', '何が分かると原因追える？', 'どの値を見せればいい？')[case_index % 4]}", clarifications[variants]),
        (f"{p} {('短く言うと何する？', '結局どこから？', 'まず一言で言うと？', '一旦なにを試せばいい？')[case_index % 4]}", summaries[variants]),
    ]


def make_row(case_index: int, intent_index: int, case: dict[str, str]) -> dict:
    context, answer = render_case(case, case_index)[intent_index]
    answer = finish(answer)
    if (case_index + intent_index * 3) % 11 == 0:
        answer = answer.replace("。", "。\n", 1)
    scenario_id = f"synth-case-{case_index:03d}"
    sample_id = f"{scenario_id}-{INTENTS[intent_index]}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "message_ids": [sample_id],
            "context_message_ids": [],
            "session_id": sample_id,
            "synthetic_scenario_id": scenario_id,
            "intent": INTENTS[intent_index],
            "category": f"synthetic_{case['topic']}_{INTENTS[intent_index]}",
            "target_post_count": 1,
            "target_chars": len(answer),
            "low_weight_review": False,
            "synthetic": True,
            "provenance": "assistant_authored_generic_scenario",
        },
    }


def build_rows(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    cases = load_cases()
    order = list(range(len(cases)))
    random.Random(seed).shuffle(order)
    validation_cases = set(order[:VALIDATION_SCENARIOS])
    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    for case_index, case in enumerate(cases, start=1):
        destination = validation_rows if case_index - 1 in validation_cases else train_rows
        destination.extend(make_row(case_index, intent_index, case) for intent_index in range(ROWS_PER_CASE))

    random.Random(seed + 1).shuffle(train_rows)
    random.Random(seed + 2).shuffle(validation_rows)

    rows = train_rows + validation_rows
    contexts = [row["messages"][1]["content"] for row in rows]
    answers = [row["messages"][2]["content"] for row in rows]
    if len(rows) != 2_000:
        raise SystemExit(f"Expected 2,000 examples, found {len(rows)}")
    if len(train_rows) != 1_800 or len(validation_rows) != 200:
        raise SystemExit("Expected a scenario-held-out split of 1,800 train / 200 validation")
    if len(set(contexts)) != len(contexts):
        raise SystemExit("Synthetic dataset contains duplicate contexts")
    if len(set(answers)) != len(answers):
        raise SystemExit("Synthetic dataset contains duplicate target replies")
    if {row["metadata"]["synthetic_scenario_id"] for row in train_rows} & {
        row["metadata"]["synthetic_scenario_id"] for row in validation_rows
    }:
        raise SystemExit("A scenario appears in both train and validation")
    return train_rows, validation_rows, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    train_rows, validation_rows, rows = build_rows(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_rows in (
        ("synthetic_explanations_train.jsonl", train_rows),
        ("synthetic_explanations_validation.jsonl", validation_rows),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    case_topics = Counter(case["topic"] for case in load_cases())
    categories = Counter(row["metadata"]["category"] for row in rows)
    targets_by_intent = {
        intent: [row["metadata"]["target_chars"] for row in rows if row["metadata"]["intent"] == intent]
        for intent in INTENTS
    }
    target_lengths = [row["metadata"]["target_chars"] for row in rows]
    manifest = {
        "author": "assistant-authored, owner-directed revision",
        "status": "synthetic examples; review before training; not autobiographical evidence",
        "source": "Observed broad topic and style cues from the local Discord corpus; no Discord message text copied",
        "purpose": "Add specific, useful answers to explanation and help requests while preserving concise chat replies",
        "examples": len(rows),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "authored_scenarios": len(load_cases()),
        "intents_per_scenario": list(INTENTS),
        "rows_per_scenario": ROWS_PER_CASE,
        "validation_scenarios_held_out_whole": VALIDATION_SCENARIOS,
        "unique_contexts": len({row["messages"][1]["content"] for row in rows}),
        "unique_target_replies": len({row["messages"][2]["content"] for row in rows}),
        "target_reply_chars": {
            "count": len(target_lengths),
            "median": median(target_lengths),
            "mean": round(mean(target_lengths), 1),
            "min": min(target_lengths),
            "max": max(target_lengths),
        },
        "target_reply_chars_by_intent": {
            intent: {
                "count": len(lengths),
                "median": median(lengths),
                "mean": round(mean(lengths), 1),
                "min": min(lengths),
                "max": max(lengths),
            }
            for intent, lengths in targets_by_intent.items()
        },
        "topics": dict(sorted(case_topics.items())),
        "categories": dict(sorted(categories.items())),
        "style_basis": [
            "informal Japanese with natural reactions and questions",
            "short answers for simple requests; concrete detail when asked",
            "occasional hedging and line breaks, without fixed catchphrases",
        ],
        "guardrails": [
            "Scenarios are generic and do not assert Kobajin's personal experiences, preferences, or beliefs.",
            "No private Discord excerpts, names, handles, contact details, or message IDs are included.",
            "Validation scenarios are held out as complete groups and must not be used for training.",
        ],
    }
    (args.output_dir / "synthetic_explanations_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
