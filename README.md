# Face Tell

**Live: [facetell.vercel.app](https://facetell.vercel.app)**

Real-time facial expression recognition and nervousness detection from a webcam,
running in the browser.

Two approaches were built and measured against each other: a lightweight
classifier over MediaPipe's 52 facial-muscle scores, and a fine-tuned
EfficientNet CNN over raw pixels. On top of either sits a temporal layer that
estimates nervousness from blink rate, gaze instability and expression churn —
because nervousness is not an expression, it is a pattern over time.

Everything here was measured rather than assumed. Where a decision turned out
to be wrong, the wrong version and its number are written down too.

---

## How it runs

| stage | where | what it does |
|---|---|---|
| face tracking | your browser | MediaPipe finds the face and reports 52 muscle scores |
| expression | your browser | fine-tuned EfficientNet-B0 over ONNX Runtime, ~29ms |
| nervousness | your browser | arithmetic over time against your own baseline |
| description | Modal (cloud GPU) | LoRA-tuned Llama-3.2-3B turns the numbers into a sentence |

Only the describer leaves the machine, and it receives **52 numbers, never an
image**. No video frame is ever uploaded, so "your camera never leaves your
device" is literal rather than marketing.

The GPU scales to zero, so the first description after an idle period waits
~30s for a container; subsequent ones take ~3s.

## Results

All figures are from held-out test splits, logged in [`runs/`](runs/).

| model | trained on | accuracy | macro recall |
|---|---|---|---|
| blendshapes + gradient boosting | RAF-DB | 73.8% | 55.6% |
| blendshapes + gradient boosting | AffectNet | 56.3% | 53.6% |
| fine-tuned EfficientNet-B0 | RAF-DB | **90.6%** | **82.7%** |
| fine-tuned EfficientNet-B0 | AffectNet | 75.3% | 72.8% |
| fine-tuned EfficientNet-B0 | both, merged | 79.5% | 76.7% |

**Macro recall is the number to read.** Plain accuracy is dominated by whichever
classes happen to be common; on RAF-DB, 57% of images are happy or neutral, so a
model can score well while being useless at fear.

**The RAF-DB model is the one deployed**, and the reason is worth stating.
The merged model generalises better on paper — see
[cross-dataset results](#the-cross-dataset-result) — but tested side by side on
an actual webcam, the RAF-DB one read faces noticeably better.

That is not a contradiction. A 640×480 frame in a dim room, cropped to ~200px,
looks like RAF-DB: soft, low-detail, upscaled from a small original. AffectNet
is sharp 1547px photography, which no webcam resembles. The merged model spent
half its capacity learning a domain this app never sees.

So "best model" depends on where it runs. On a better camera the ranking would
likely flip. The honest framing is that this deployment is tuned to a webcam,
not that 90.6% is the truer number.

### Pixels versus blendshapes, per class

Both trained and tested on RAF-DB:

| class | blendshapes | CNN | change |
|---|---|---|---|
| neutral | 94% | 100% | +6 |
| happy | 85% | 93% | +8 |
| angry | 68% | 76% | +8 |
| surprised | 65% | 88% | +23 |
| **sad** | 41% | 83% | **+42** |
| **disgusted** | 21% | 62% | **+41** |
| **fear** | 16% | 60% | **+44** |

The classes blendshapes already handled gained ~8 points. The three it *failed*
at gained 41–44. Compressing a face to 52 muscle sliders discards exactly the
detail that separates disgust from anger and fear from surprise — the crease at
the nose bridge, the curl of the upper lip. A CNN sees those pixels.

---

## The cross-dataset result

A held-out split only proves a model works on *more photos from the same
collection*. The honest test is a different collection entirely.

| model | own test set | different dataset |
|---|---|---|
| blendshapes (RAF-DB) | 73.8% | **33.4%** |
| CNN (AffectNet) | 75.3% | **~44%** |

Roughly half the performance disappears. The blendshape model collapsed almost
completely — it predicted `neutral` for nearly everything, because MediaPipe
reads weaker muscle activations off RAF-DB's soft 100px images than off
AffectNet's sharp ones, so every learned threshold landed in the wrong place.
Blendshapes are person-independent. They are not capture-independent.

Investigating that gap turned up two problems in the source data:

**5,130 byte-identical images appear in both datasets**, all in `neutral`. One
of the HuggingFace re-uploads padded its neutral class from the other. Their
apparent 96% "cross-dataset transfer" was the model recognising images it had
trained on. Excluding them, honest cross-dataset macro recall is 42.8%, not 50.4%.

**The datasets disagree about what `happy` means.** AffectNet's happy is a mild,
closed-mouth adult smile; RAF-DB's is a wide toothy laugh, often a child's
(`mouthStretch` is 2.8× higher). A model trained on AffectNet scored **0.1% on
RAF-DB's happy faces** — 5 correct out of 5,197 — calling them angry, disgusted
or afraid. Bared teeth and a stretched mouth mean something different in each
collection.

**The fix was data, not architecture.** Merging both datasets, dropping the
duplicates and retraining with identical hyperparameters took happy from
**0.1% → 94%**. The merged model also beats the AffectNet specialist on
AffectNet's own test set (76.0% vs 75.3%) while giving up 3 points on RAF-DB.

Also found: **54 AffectNet images carry two different labels** — the same
photograph filed as both `angry` and `disgusted`, or `disgusted` and `fear`.
A useful reminder of why published AffectNet results plateau around 65%.

---

## Quick start

```bash
py -3.12 -m venv .venv          # 3.12 specifically: MediaPipe has no 3.13+ wheels
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

curl.exe -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

python webcam_test.py           # check the camera and face tracking work
```

For GPU training, install PyTorch separately:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Reproducing the results

```bash
# 1. Extract blendshapes and resized JPEGs from both datasets (~15 min each)
python extract.py affectnet --save-images
python extract.py rafdb --save-images

# 2. Merge them, dropping the duplicated images
python merge_datasets.py

# 3. Baseline: classifier over the 52 blendshape scores
python train.py --csv data/affectnet_blendshapes.csv --out models/affectnet_blendshape.joblib

# 4. Fine-tune the CNN (~30 min on an RTX 3050)
python finetune.py --dataset combined --model efficientnet_b0 --epochs 25 \
                   --aug strong --lr 1e-3 --class-weights --tag combined_best

# 5. Test it on a dataset it never trained on
python cross_eval.py --checkpoint models/finetuned_combined_best.pt --on rafdb

# 6. Run it live
python live_cnn.py --checkpoint models/finetuned_combined_best.pt
```

Datasets download automatically from HuggingFace. A free `hf auth login` token
makes this several times faster — unauthenticated streaming is heavily throttled.

---

## Hyperparameter sweep

`sweep.py` varies one factor at a time against a fixed reference config and
ranks by macro recall. Full logs in [`runs/`](runs/).

| config | accuracy | macro |
|---|---|---|
| `aug=strong` | 88.8% | **81.8%** |
| `lr=1e-3` | 88.5% | 80.7% |
| `aug=medium` (reference) | 88.4% | 80.2% |
| `aug=light` | 87.2% | 79.2% |
| `lr=1e-4` | 84.8% | 79.0% |

What it actually showed:

- **Learning rate is a threshold, not a dial.** 3e-4 and 1e-3 differ by three
  images out of 2,745 — a tie. Drop to 1e-4 and you lose 102 images. And 1e-4
  was still improving at the final epoch: undertrained, not worse.
- **Strong augmentation wins on the hard classes.** It took the train/val gap
  from 8.7 points to 3.0, and beat medium on disgust, fear, sad and surprised.
- **Combining both findings** (`aug=strong`, `lr=1e-3`, 30 epochs) gave 90.6% /
  82.7% — the best single-dataset result here.

---

## The describer

`describe.py` -> `train_describe.py` -> `modal_describe.py`

The classifier answers with one word. A LoRA fine-tune of Llama-3.2-3B turns the
same measurements into a sentence.

**The dataset had to be invented.** No image-to-sentence dataset exists for
facial expressions, so descriptions were generated from measurements already
taken - every clause traces back to a number. Intensity bands come from the
observed value distribution (69% of readings under 0.05, p90 0.32, p95 0.53,
p99 0.84), so "faintly" and "strongly" mean something measured.

Two curation passes, both prompted by reading the output:

- gaze direction says where attention is, not what is felt, so `eyeLook*` is
  demoted and only surfaces when nothing else is active
- 6,620 rows (14.4%) dropped where the measurement flatly contradicted the
  label, e.g. `mouthSmile 0.95` on an image labelled fear

**Training:** 3,990 balanced examples, 28 min on an RTX 3050, peak 3.2GB VRAM.
24.3M of 3.2B parameters trained (0.75%). Loss 3.25 -> 0.29, converged by step
100 - about 1,600 examples would have been enough.

**It describes muscles well and infers emotion badly.** Given `browDown 0.74,
mouthPress 0.48, noseSneer 0.21` it produced a perfect description and then
called it *sad*. The CNN is 90.6% at that job, so the server splits the output
and the site shows the description with the CNN's label. The model's own verdict
is returned in the JSON but never displayed.

```
before fine-tuning:  "Based on the facial muscle activations, this person is
                      making a slight, subtle smile..."       (mouthSmile 0.88)
after:               "The mouth corners pulled up very strongly, the eyes
                      narrowed strongly."
```

The base model also called `0.33` a "high activation" and `0.41` a "lower" one.
It knew English; it had no idea what the numbers meant.

## Nervousness detection

`nervous.py` is not a model. There is no face you can pull that means "nervous";
it is a pattern over time, so it is arithmetic on four signals measured against
*your own* 30-second baseline:

| signal | weight | source |
|---|---|---|
| blink rate | 40% | `eyeBlink` blendshapes, edge-triggered with hysteresis |
| gaze instability | 25% | variance across the 8 `eyeLook` blendshapes |
| expression churn | 20% | how often the predicted label flips |
| tension expressions | 15% | sustained angry / sad / surprised probability |

Calibrating per person matters. Textbook resting blink rate is 15–20/min, but
MediaPipe peaked at 0.706 on this webcam, not 1.0 — a hardcoded 0.9 threshold
would detect zero blinks. Measured baselines across two runs on the same person
were 9/min and 22/min, so the score is comparable *within* a session, not across
sessions.

Measured: calm 15–20, acting nervous 70.

---

## Files

| file | purpose |
|---|---|
| `webcam_test.py` | camera + face landmark smoke test |
| `cam_check.py` | diagnose which OpenCV camera backend works |
| `probe_datasets.py` | measure MediaPipe detection rate per dataset before downloading |
| `extract.py` | download a dataset → blendshape CSV + resized JPEGs |
| `merge_datasets.py` | combine datasets, hash-deduplicate, report conflicts |
| `collect.py` | record your own face into a labelled CSV |
| `train.py` | train a classifier on blendshape features |
| `finetune.py` | fine-tune a pretrained CNN (the transfer learning) |
| `sweep.py` | one-factor-at-a-time hyperparameter sweep |
| `cross_eval.py` | evaluate a model on a dataset it never trained on |
| `live.py` | webcam + blendshape classifier |
| `live_cnn.py` | webcam + fine-tuned CNN, with the baseline side by side |
| `nervous.py` | temporal nervousness estimation |
| `export_onnx.py` | PyTorch → ONNX, quantise, and score each variant |
| `describe.py` | build the description dataset from measured blendshapes |
| `train_describe.py` | LoRA fine-tune the describer, with before/after samples |
| `describe_server.py` | serve the describer locally, for `live_cnn.py` |
| `modal_describe.py` | serve the describer on a cloud GPU, for the website |
| `web/` | the deployed app: HTML, ONNX model, icons, Vercel headers |

Two virtual environments, deliberately. `.venv` holds the vision stack;
`.venv-llm` holds Unsloth, which resolves its own torch build and would
otherwise replace the CUDA one the vision pipeline needs. Installing Unsloth
into `.venv` silently swapped in a CPU-only torch on the first attempt.

---

## Notes and caveats

**MediaPipe 1.0 removed `mp.solutions`.** Every FER tutorial online uses it.
This code uses the `tasks` API throughout.

**The crop fed to the CNN was measured, not guessed.** Across 199 RAF-DB images
the training frame is 0.97× the MediaPipe landmark box — the 478 landmarks
already reach the hairline, so the landmark box *is* the whole head. An earlier
0.25 margin fed the model a 25%-too-wide view and noticeably hurt live accuracy.

**Webcam accuracy is lower than the test numbers.** The training images are
internet photographs: bright, sharp, mostly frontal. A dim room and one camera
are a different distribution.

**`Piro17/affectnethq` is a curated 27,823-image subset**, not the full ~440k
AffectNet benchmark, so its numbers are not directly comparable to published
AffectNet results. RAF-DB figures are roughly comparable (literature: ~88%).

**Dataset licensing.** RAF-DB and AffectNet are normally distributed under
academic licences signed with their authors. The copies used here are
third-party HuggingFace re-uploads, which likely do not hold redistribution
rights. Fine for learning; obtain them properly for anything else.

**These are photographs of real people** scraped from the internet who did not
consent to being in a training set. `data/` is gitignored for that reason, and
no face data is committed to this repo.

---

## Deployment notes

**INT8 quantisation was measured and rejected.** The usual claim is 4x smaller
for ~1% accuracy; on EfficientNet it cost **10.5 points** (88.2% -> 77.8%) with
no speedup. Depthwise-separable convolutions have wide per-channel ranges and
squeeze-excitation blocks are rounding-sensitive. float32 ships.

**`vercel.json` sets COOP/COEP**, which is what allows multi-threaded WASM.
Without those headers onnxruntime-web silently falls back to one thread.

**The crop geometry was measured, not guessed.** Across 199 RAF-DB images the
training frame is 0.97x the MediaPipe landmark box - the 478 landmarks already
reach the hairline. An earlier 0.25 margin fed the model a 25%-too-wide view and
noticeably hurt live accuracy.

**`disgusted` is hidden on the website**, masked at inference by setting its
logit to -inf so the remaining six renormalise. Display-only; the weights are
untouched and `live_cnn.py` still shows all seven.

## Still to do

- Retrain the describer with the classifier's label in the prompt, so it stops
  guessing the emotion it is bad at
- Collect faces beyond two datasets; cross-dataset accuracy is still ~44%
- A smaller description model that could run in the browser, removing the last
  network dependency
