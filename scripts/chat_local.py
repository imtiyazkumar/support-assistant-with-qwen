"""Chat with the selected local model. Use /clear or /quit."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.inference.runtime import LocalModel  # noqa: E402
from backend.services.yember_knowledge import YemberKnowledgeBase  # noqa: E402


SUPPORT_SYSTEM = (
    "You are Yember's customer and partner support assistant. Be concise and "
    "helpful. Use only facts in the supplied Yember FAQ context. Do not invent "
    "prices, policies, capabilities, statuses, or steps. If the context does "
    "not answer the question, say that the available Yember support information "
    "does not specify it and suggest contacting Yember support or an "
    "administrator. For ordinary customer and partner questions, give clear "
    "in-app guidance and translate backend details into plain language. Never "
    "ask them to call API endpoints. Include endpoints or implementation details "
    "only when the user explicitly asks a technical question. Preserve the "
    "actors, action order, and important conditions stated in the FAQ; do not "
    "assign one role's action to another role. Do not guess outside the FAQ."
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = LocalModel()
    knowledge = YemberKnowledgeBase()
    history: list[dict[str, str]] = []
    print("Yember support assistant (local Qwen3.5-2B).")
    print("Answers use the local Yember FAQ. Commands: /clear, /quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/quit":
            return
        if question == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if not question:
            continue
        previous_user = next(
            (
                message["content"]
                for message in reversed(history)
                if message["role"] == "user"
            ),
            None,
        )
        retrieval_query = (
            f"{previous_user} {question}"
            if previous_user and len(question.split()) <= 4
            else question
        )
        context, source_question = knowledge.answer_context(retrieval_query)
        if context is None:
            words = set(question.lower().strip(" .!?\t").split())
            if words and words <= {
                "hi",
                "hello",
                "hey",
                "there",
                "yember",
                "thanks",
                "thank",
                "you",
            }:
                print(
                    "\nAssistant: Hi! I can help with Yember customer, "
                    "restaurant, and delivery questions."
                )
            else:
                print(
                    "\nAssistant: I can't confirm that from the available Yember "
                    "support information. Please contact Yember support or an "
                    "administrator."
                )
            continue
        candidate = [
            {"role": "system", "content": SUPPORT_SYSTEM + "\n\n" + context},
            *history,
            {"role": "user", "content": question},
        ]
        try:
            result = runtime.generate(candidate)
        except ValueError as exc:
            print(str(exc))
            continue
        print("\nAssistant:", result["answer"])
        if source_question:
            print(f"[Yember FAQ: {source_question}]")
        if result["output_truncated"]:
            print("[Output token limit reached.]")
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]


if __name__ == "__main__":
    main()
