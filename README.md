# FaceTell

Real-time facial expression recognition and nervousness detection from a webcam.

Two approaches were built and measured against each other: a lightweight
classifier over MediaPipe's 52 facial-muscle scores, and a fine-tuned
EfficientNet CNN over raw pixels. On top of either sits a temporal layer that
estimates nervousness from blink rate, gaze instability and expression churn —
because nervousness is not an expression, it is a pattern over time.

Everything here was measured rather than assumed. Where a decision turned out
to be wrong, the wrong version and its number are written down too.

---

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

**The 79.5% model is the one that ships**, despite the 90.6% one existing. See
[cross-dataset results](#the-cross-dataset-result) below for why.

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

## Still to do

- Export to ONNX and quantise to INT8 for browser deployment
- Client-side web app (`onnxruntime-web`) so video never leaves the user's machine
- Re-wire `nervous.py` onto the CNN instead of the blendshape model
- Optional: LoRA fine-tune a small vision-language model to describe the
  expression in words rather than emit a label
