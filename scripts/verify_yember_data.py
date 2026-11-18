"""Validate the Yember chat-format dataset without loading the model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROLES = ["system", "user", "assistant"]
SECRET_PATTERNS = (
    re.compile(r"(?i)\bsk_(?:live|test)_[a-z0-9]{16,}\b"),
    re.compile(r"(?i)\b(?:password|api[_ -]?key)\s*[:=]\s*['\"]?[^\s,;]{8,}"),
    re.compile(r"\b\d{10}\b"),
)


def validate(path: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        messages = value.get("messages") if isinstance(value, dict) else None
        if (
            not isinstance(messages, list)
            or any(not isinstance(message, dict) for message in messages)
            or [message.get("role") for message in messages] != EXPECTED_ROLES
        ):
            errors.append(f"line {line_number}: expected system/user/assistant messages")
            continue
        contents = [message.get("content") for message in messages]
        if any(not isinstance(content, str) or not content.strip() for content in contents):
            errors.append(
                f"line {line_number}: message content must be non-empty text"
            )
            continue
        rows.append(dict(zip(EXPECTED_ROLES, contents)))

    questions = [row["user"].strip().casefold() for row in rows]
    if len(set(questions)) != len(questions):
        errors.append("duplicate user questions found")
    systems = {row["system"] for row in rows}
    if len(systems) > 1:
        errors.append("system instructions differ between examples")
    for number, row in enumerate(rows, 1):
        if len(row["assistant"]) > 5000:
            errors.append(f"record {number}: answer exceeds 5,000 characters")
        for pattern in SECRET_PATTERNS:
            if any(pattern.search(row[role]) for role in ("user", "assistant")):
                errors.append(f"record {number}: possible credential or phone number")
                break
    if not rows:
        errors.append("dataset contains no valid records")
    return {
        "status": "passed" if not errors else "failed",
        "records": len(rows),
        "unique_questions": len(set(questions)),
        "average_answer_characters": (
            round(sum(len(row["assistant"]) for row in rows) / len(rows))
            if rows
            else 0
        ),
        "errors": errors,
        "note": (
            "Structural checks cannot confirm that Yember product facts match "
            "the live application."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/train.jsonl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.data)
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
