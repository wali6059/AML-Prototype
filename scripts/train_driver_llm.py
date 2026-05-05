from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.config import ARTIFACT_DIR, ensure_directories

EXPERIMENT_DIR = ARTIFACT_DIR / "experiments"


class ChatRows(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.items = []
        for row in rows:
            text = format_chat(row["messages"], tokenizer)
            tokens = tokenizer(text, truncation=True, max_length=max_length)
            self.items.append(tokens["input_ids"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return torch.tensor(self.items[index], dtype=torch.long)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_chat(messages, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for msg in messages:
        role = msg["role"].title()
        parts.append(f"{role}: {msg['content']}")
    return "\n".join(parts) + "\n"


def collate(batch, tokenizer):
    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    mask = (padded != tokenizer.pad_token_id).long()
    labels = padded.clone()
    labels[mask == 0] = -100
    return {"input_ids": padded, "attention_mask": mask, "labels": labels}


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    return float(sum(losses) / max(len(losses), 1))


def generate_answers(model, tokenizer, rows, device, max_new_tokens: int = 120) -> list[dict]:
    model.eval()
    out = []
    for row in rows[:8]:
        messages = row["messages"][:-1]
        expected = row["messages"][-1]["content"]
        prompt = format_chat(messages, tokenizer)
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True)
        zone = messages[-1]["content"].split(" in ")[-1].split(".")[0]
        out.append(
            {
                "question": messages[-1]["content"],
                "expected": expected,
                "generated": text.strip(),
                "mentions_zone": int(zone in text),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "driver_llm"))
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="")
    args = parser.parse_args()

    ensure_directories()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    train_rows = read_jsonl(EXPERIMENT_DIR / "llm_train.jsonl")
    eval_rows = read_jsonl(EXPERIMENT_DIR / "llm_eval.jsonl")
    train_ds = ChatRows(train_rows, tokenizer, args.max_length)
    eval_ds = ChatRows(eval_rows, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        eval_loss = evaluate_loss(model, eval_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": float(sum(losses) / max(len(losses), 1)), "eval_loss": eval_loss})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    generations = generate_answers(model, tokenizer, eval_rows, device)
    metrics = {
        "base_model": args.model,
        "device": str(device),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "epochs": args.epochs,
        "final_train_loss": history[-1]["train_loss"],
        "final_eval_loss": history[-1]["eval_loss"],
        "eval_perplexity": float(math.exp(min(history[-1]["eval_loss"], 20))),
        "zone_mention_rate": float(sum(row["mentions_zone"] for row in generations) / max(len(generations), 1)),
        "output_dir": str(output_dir),
    }
    if args.push_to_hub:
        repo_id = args.hub_model_id or "wali6059/tip-or-skip-driver-llm"
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        metrics["hub_model_id"] = repo_id
    pd.DataFrame(history).to_csv(EXPERIMENT_DIR / "llm_training_history.csv", index=False)
    pd.DataFrame(generations).to_csv(EXPERIMENT_DIR / "llm_generation_eval.csv", index=False)
    (EXPERIMENT_DIR / "llm_finetune_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
