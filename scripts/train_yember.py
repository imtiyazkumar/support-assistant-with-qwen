"""Fine-tune a small QLoRA adapter from docs/train.jsonl."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.environment import configure_environment  # noqa: E402
from scripts.verify_yember_data import validate  # noqa: E402


def load_examples(
    path: Path, tokenizer: Any, validation_fraction: float
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = [json.loads(line)["messages"] for line in path.read_text().splitlines()]
    examples = [
        {"text": tokenizer.apply_chat_template(row, tokenize=False)}
        for row in messages
    ]
    indices = list(range(len(examples)))
    random.Random(42).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation_ids = set(indices[:validation_count])
    train = [example for i, example in enumerate(examples) if i not in validation_ids]
    validation = [example for i, example in enumerate(examples) if i in validation_ids]
    return train, validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/train.jsonl")
    parser.add_argument(
        "--base-model", type=Path, default=ROOT / "models/base/Qwen3.5-2B"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models/adapters/yember-support"
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check data/model files without starting GPU training",
    )
    args = parser.parse_args()

    report = validate(args.data)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        return 1
    if not args.base_model.joinpath("config.json").is_file():
        parser.error(f"Base model files are missing: {args.base_model}")
    if not 0.05 <= args.validation_fraction <= 0.3:
        parser.error("--validation-fraction must be between 0.05 and 0.3")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.validate_only:
        print("Dataset and local base model are present. GPU training was not run.")
        return 0

    configure_environment()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Yember QLoRA training requires a BF16 capable CUDA GPU"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, validation_rows = load_examples(
        args.data, tokenizer, args.validation_fraction
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    adapter = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(ROOT / "training/checkpoints/yember-support"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=512,
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=5,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        peft_config=adapter,
    )
    result = trainer.train()
    evaluation = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(
        json.dumps(
            {
                "adapter": str(args.output),
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "training_loss": result.training_loss,
                "evaluation_loss": evaluation.get("eval_loss"),
                "global_steps": result.global_step,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
