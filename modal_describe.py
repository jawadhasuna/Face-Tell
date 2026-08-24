"""Serve the fine-tuned describer on Modal, so the deployed site can use it.

The browser keeps doing the seeing: MediaPipe finds the face and the CNN
classifies it, both locally. Only the 52 muscle numbers are sent here -
never an image, never a video frame - so "your camera never leaves your
device" stays literally true.

Scale-to-zero: no GPU runs while nobody is using the page. The first
request after an idle period pays a cold start; scaledown_window keeps a
container alive for a few minutes after the last one so a person trying
several expressions only waits once.

    modal setup                      # once, to authenticate
    modal deploy modal_describe.py   # prints the public URL
    modal app logs facetell-describe

The adapter is baked into the image from ./lora/describe, so redeploy
after retraining.
"""

import re

import modal

APP_NAME = "facetell-describe"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MINUTES = 60

SYSTEM = "You read facial muscle measurements and describe the expression in plain language."
ASK = "What expression is this person making?"

# The model reliably ends with a verdict sentence ("The overall read is sad.").
# Its muscle description is trustworthy; its emotion guess is not - the CNN is
# 90.6% on that and the LLM saw only eight numbers. Split them so the caller
# can keep the description and supply its own label.
VERDICT = re.compile(
    r"\s*((?:this\s+)?(?:reads?\s+as|the\s+overall\s+read\s+is|the\s+face\s+is|"
    r"this\s+is|no\s+strong)[^.]*\.)\s*$", re.IGNORECASE)


def split_verdict(text: str) -> tuple[str, str]:
    """Return (muscle description, the model's own verdict sentence)."""
    match = VERDICT.search(text.strip())
    if not match:
        return text.strip(), ""
    return text[:match.start()].strip(), match.group(1).strip()


# Trailing intensity adverb, so "the eyes narrowed strongly" and "the eyes
# narrowed faintly" can be recognised as the same body part described twice.
INTENSITY = re.compile(
    r"\s+(?:very\s+strongly|strongly|clearly|faintly)\s*$", re.IGNORECASE)


def dedupe_clauses(description: str) -> str:
    """Drop clauses describing a body part the sentence already covered.

    The model sometimes reuses a phrase instead of moving to the next signal
    ("the eyes narrowed strongly, the eyes narrowed faintly"). Keeping the
    first mention preserves the stronger reading.
    """
    prefix, sep, rest = description.partition(", ")
    if not sep:
        return description

    clauses = [prefix] + rest.split(", ")
    seen, kept = set(), []
    for clause in clauses:
        # Strip the sentence-final full stop before the adverb, or the anchored
        # pattern misses the last clause and it never matches its twin.
        key = INTENSITY.sub("", clause.strip().rstrip(".").strip()).lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(clause.strip().rstrip(".").strip())

    joined = ", ".join(kept)
    if not joined:
        return description
    return joined[0].upper() + joined[1:] + "."


# Plain transformers + peft rather than Unsloth: inference does not need the
# training kernels, and a smaller image means a faster cold start.
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned to the versions that produced the adapter. Its config declares
    # peft_version 0.20.0 and carries fields (arrow_config, qalora_group_size,
    # target_parameters) that older peft does not know about, so an older
    # image would fail to load it.
    .pip_install(
        "torch==2.11.0",
        "transformers==5.5.0",
        "peft==0.20.0",
        "bitsandbytes==0.50.1",
        "accelerate==1.14.0",
        "sentencepiece==0.2.2",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir("lora/describe", remote_path="/adapter", copy=True)
)

# Cache the base weights between cold starts instead of re-downloading 2GB.
hf_cache = modal.Volume.from_name("facetell-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="T4",                      # 16GB, cheapest tier that fits a 3B in 4-bit
    volumes={"/cache": hf_cache},
    scaledown_window=5 * MINUTES,  # stay warm briefly so repeat requests are fast
    timeout=10 * MINUTES,
)
class Describer:
    @modal.enter()
    def load(self):
        import os
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache"

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir="/cache")
        # No dtype argument: the repo is already 4-bit and its quantization_config
        # carries the compute dtype. transformers 5 renamed torch_dtype -> dtype,
        # so passing either is a needless version dependency.
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, cache_dir="/cache", device_map="cuda")
        self.model = PeftModel.from_pretrained(base, "/adapter").eval()
        print(f"loaded on {next(self.model.parameters()).device}", flush=True)
        hf_cache.commit()
        print("model + adapter loaded", flush=True)

    def _generate(self, readings: dict, max_new_tokens: int = 70) -> str:
        import torch

        ranked = sorted(readings.items(), key=lambda kv: -float(kv[1]))[:8]
        listing = ", ".join(f"{k} {float(v):.2f}" for k, v in ranked)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user",
              "content": f"Facial muscle activations (0-1): {listing}\n\n{ASK}"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @modal.method()
    def run(self, readings: dict) -> dict:
        """Callable with .remote() from a local entrypoint, for smoke tests."""
        full = self._generate(readings)
        description, verdict = split_verdict(full)
        return {"text": full, "description": dedupe_clauses(description),
                "model_verdict": verdict}

    @modal.fastapi_endpoint(method="POST", docs=True)
    def describe(self, payload: dict):
        """POST {"readings": {"mouthSmile": 0.88, ...}} -> {"text": "..."}"""
        readings = (payload or {}).get("readings") or {}
        if not readings:
            return {"error": "supply a non-empty 'readings' object"}
        try:
            full = self._generate(readings)
            description, verdict = split_verdict(full)
            return {"text": full, "description": dedupe_clauses(description),
                    "model_verdict": verdict}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"ok": True}


@app.local_entrypoint()
def smoke():
    """modal run modal_describe.py - check it works before deploying."""
    probes = [
        {"mouthSmile": 0.88, "eyeSquint": 0.61, "cheekSquint": 0.34},
        {"browInnerUp": 0.79, "browOuterUp": 0.71, "jawOpen": 0.55},
        {"browDown": 0.74, "mouthPress": 0.48, "eyeSquint": 0.39},
    ]
    describer = Describer()
    for readings in probes:
        result = describer.run.remote(readings)
        print(f"\nin  : {readings}")
        print(f"desc: {result['description']}")
        print(f"(its own verdict, unused by the app: {result['model_verdict']})")
