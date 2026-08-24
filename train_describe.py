"""LoRA fine-tune a small LLM to describe expressions from muscle readings.

The CNN answers with one word. This teaches a chat model to turn the same
underlying measurements into a sentence, so the app can explain itself.

Nothing here changes the base model. LoRA adds small trainable matrices
alongside the frozen weights in every attention and MLP block; only those
train, and they save out as a ~50MB adapter.

Runs in .venv-llm, not the CV environment - Unsloth pulls its own torch
and would otherwise replace the CUDA build the vision pipeline needs.

Usage:
    python train_describe.py --dry-run          # load, sample, no training
    python train_describe.py
    python train_describe.py --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent

# Held out by hand: these are the readings the samples are generated from,
# so before/after is judged on inputs the model never trained on.
PROBES = [
    "mouthSmile 0.88, eyeSquint 0.61, cheekSquint 0.34, mouthUpperUp 0.22",
    "browInnerUp 0.79, browOuterUp 0.71, jawOpen 0.55, eyeWide 0.40",
    "browDown 0.74, mouthPress 0.48, eyeSquint 0.39, noseSneer 0.21",
    "eyeBlink 0.66, eyeLookDown 0.41, mouthFrown 0.33, browInnerUp 0.19",
]
SYSTEM = "You read facial muscle measurements and describe the expression in plain language."
ASK = "What expression is this person making?"


def build_prompt(tokenizer, readings: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Facial muscle activations (0-1): {readings}\n\n{ASK}"}],
        tokenize=False, add_generation_prompt=True,
    )


def sample(model, tokenizer, title: str) -> None:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    for readings in PROBES:
        inputs = tokenizer(build_prompt(tokenizer, readings),
                           return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=70, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
        print(f"\n  in : {readings}")
        print(f"  out: {answer}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    p.add_argument("--train", default="data/describe_train.jsonl")
    p.add_argument("--out", default="lora/describe")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--dry-run", action="store_true",
                   help="load the model and show untrained output, then stop")
    args = p.parse_args()

    from unsloth import FastLanguageModel  # must import before transformers
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    print(f"loading {args.model} in 4-bit...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,          # let Unsloth pick bf16 on Ampere
    )

    sample(model, tokenizer, "BEFORE — the base model, untrained on this task")
    if args.dry_run:
        return

    # LoRA attaches to attention and MLP projections in every block, not to a
    # final layer: the frozen weights stay, small matrices are added alongside.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\ntraining {trainable/1e6:.1f}M of {total/1e6:.0f}M parameters "
          f"({trainable/total:.2%})\n", flush=True)

    data = load_dataset("json", data_files=str(ROOT / args.train), split="train")

    # This TRL build will not auto-detect the conversational `messages` format,
    # so render each conversation to a single string with the model's own chat
    # template and train on that.
    def render(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)}

    data = data.map(render, remove_columns=data.column_names)
    print(f"{len(data)} training examples", flush=True)
    print(f"\nfirst rendered example:\n{'-' * 60}\n{data[0]['text']}\n{'-' * 60}\n", flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=0,
            output_dir="outputs",
            report_to="none",
            max_length=args.max_seq,
            dataset_text_field="text",
            dataset_num_proc=1,        # Windows: >1 spawns processes that hang
        ),
    )

    stats = trainer.train()
    peak = torch.cuda.max_memory_reserved() / 1e9
    print(f"\ntrained in {stats.metrics['train_runtime']/60:.1f} min, "
          f"peak VRAM {peak:.1f} GB")

    out_dir = ROOT / args.out
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"adapter saved to {out_dir} ({size:.0f} MB)")

    sample(model, tokenizer, "AFTER — same inputs, same questions")

    (out_dir / "training_summary.json").write_text(json.dumps({
        "base_model": args.model,
        "examples": len(data),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "learning_rate": args.lr,
        "trainable_params": trainable,
        "total_params": total,
        "train_runtime_min": round(stats.metrics["train_runtime"] / 60, 2),
        "final_loss": stats.metrics.get("train_loss"),
        "peak_vram_gb": round(peak, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
