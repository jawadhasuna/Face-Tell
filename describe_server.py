"""Serve the fine-tuned describer over HTTP, on localhost only.

The camera app lives in .venv (MediaPipe, OpenCV) and this lives in
.venv-llm (Unsloth, transformers). They cannot share a process, so the
model sits behind a tiny local endpoint instead. Nothing is exposed
beyond 127.0.0.1.

    POST /describe   {"readings": {"mouthSmile": 0.88, ...}}
    -> {"text": "The mouth corners pulled up very strongly, ..."}

Run in .venv-llm:
    python describe_server.py
    python describe_server.py --base-only     # compare against untrained
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent

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


model = tokenizer = None


def format_readings(readings: dict, top: int = 8) -> str:
    ranked = sorted(readings.items(), key=lambda kv: -float(kv[1]))[:top]
    return ", ".join(f"{k} {float(v):.2f}" for k, v in ranked)


def generate(readings: dict, max_new_tokens: int = 70) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user",
          "content": f"Facial muscle activations (0-1): {format_readings(readings)}\n\n{ASK}"}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The browser app is served from another local port.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "POST /describe"})

    def do_POST(self):
        if self.path != "/describe":
            self._send(404, {"error": "POST /describe"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            readings = payload.get("readings") or {}
            if not readings:
                self._send(400, {"error": "no readings supplied"})
                return
            full = generate(readings)
            description, verdict = split_verdict(full)
            self._send(200, {"text": full, "description": description,
                             "model_verdict": verdict})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *_):
        pass   # the default logger writes a line per request to stderr


def main() -> None:
    global model, tokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="lora/describe")
    p.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    p.add_argument("--base-only", action="store_true",
                   help="skip the adapter, to hear the untrained model")
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--max-seq", type=int, default=512)
    args = p.parse_args()

    from unsloth import FastLanguageModel

    adapter = ROOT / args.adapter
    source = args.base if args.base_only else str(adapter)
    if not args.base_only and not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}. Run train_describe.py first, "
                         f"or pass --base-only.")

    print(f"loading {source}...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)

    warm = generate({"mouthSmile": 0.9, "eyeSquint": 0.5})
    print(f"warmup: {warm}\n")

    print(f"listening on http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
