"""Small, dependency-free retrieval over the curated Yember support FAQ."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from backend.services.environment import ROOT

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    "a an and are as at be can do does for from how i if in is it me my much need new not of on"
    " or say says happen happens the their them this to was we what when where"
    " step steps which who why with you your"
    .split()
)
_CANONICAL = {
    "confirmed": "status",
    "confirmation": "status",
    "confirm": "status",
    "razorpay": "online",
    "opening": "operating",
    "onbard": "onboarding",
    "onboard": "onboarding",
    "onboarded": "onboarding",
}


def _terms(text: str, *, expand_intents: bool = False) -> set[str]:
    terms = set()
    raw_tokens = _TOKEN.findall(text.lower())
    for token in raw_tokens:
        if len(token) > 1 and token not in _STOP_WORDS:
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            token = _CANONICAL.get(token, token)
            terms.add(token)
    onboarding_intent = expand_intents and (
        "restaurant" in terms
        and "menu" not in terms
        and bool(
            set(raw_tokens)
            & {"add", "create", "register", "onbard", "onboard", "onboarding"}
        )
    )
    if onboarding_intent:
        terms.discard("add")
        terms.add("onboarding")
    elif "add" in terms:
        terms.discard("add")
        terms.add("create")
    return terms


@dataclass(frozen=True)
class KnowledgeEntry:
    question: str
    answer: str
    terms: frozenset[str]
    answer_terms: frozenset[str]


class YemberKnowledgeBase:
    """Rank FAQ records using inverse document frequency weighted overlap."""

    def __init__(self, path: Path | None = None) -> None:
        source = path or ROOT / "docs/train.jsonl"
        entries: list[KnowledgeEntry] = []
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            try:
                messages = json.loads(line)["messages"]
                question = messages[1]["content"].strip()
                answer = messages[2]["content"].strip()
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid Yember FAQ record on line {line_number}"
                ) from exc
            terms = _terms(question)
            answer_terms = _terms(answer)
            if not question or not answer or not terms:
                raise ValueError(f"Empty Yember FAQ record on line {line_number}")
            entries.append(
                KnowledgeEntry(question, answer, frozenset(terms), frozenset(answer_terms))
            )
        if not entries:
            raise ValueError(f"No Yember FAQ records found in {source}")
        self.entries = tuple(entries)
        self.document_frequency: dict[str, int] = {}
        for entry in self.entries:
            for term in entry.terms | entry.answer_terms:
                self.document_frequency[term] = (
                    self.document_frequency.get(term, 0) + 1
                )
        self.unknown_term_weight = math.log1p(len(self.entries)) * 4

    def search(self, query: str, *, limit: int = 1) -> list[tuple[float, KnowledgeEntry]]:
        query_terms = _terms(query, expand_intents=True)
        if not query_terms:
            return []
        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in self.entries:
            question_overlap = query_terms & entry.terms
            answer_overlap = query_terms & entry.answer_terms
            if not question_overlap and not answer_overlap:
                continue
            # Rare matching words carry more weight than broad terms like "order".
            weight = lambda term: math.log1p(  # noqa: E731
                len(self.entries) / self.document_frequency[term]
            ) if term in self.document_frequency else self.unknown_term_weight
            denominator = sum(weight(term) for term in query_terms)
            score = (
                3 * sum(weight(term) for term in question_overlap)
                + 0.5 * sum(weight(term) for term in answer_overlap - question_overlap)
            ) / (3 * denominator)
            scored.append((score, entry))
        return sorted(scored, key=lambda row: row[0], reverse=True)[:limit]

    def answer_context(self, query: str) -> tuple[str | None, str | None]:
        """Return a grounded prompt excerpt and its FAQ question, if relevant."""
        hits = self.search(query, limit=len(self.entries))
        # A strong partial match can still point to the wrong FAQ topic. Require
        # every meaningful query term to occur in this record's question/answer.
        query_terms = _terms(query, expand_intents=True)
        match = next(
            (
                (score, entry)
                for score, entry in hits
                if score >= 0.16
                and query_terms <= (entry.terms | entry.answer_terms)
            ),
            None,
        )
        if match is None:
            return None, None
        _, entry = match
        return (
            "Use this Yember FAQ excerpt to answer the user's question. "
            "Do not add details that are absent from it. If it does not answer "
            "the question, say the available Yember information does not specify "
            "the answer.\n\n"
            f"FAQ question: {entry.question}\nFAQ answer: {entry.answer}",
            entry.question,
        )
