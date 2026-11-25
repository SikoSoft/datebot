# How datebot works inside

This explains what happens between "a photo arrives" and "a score from 0 to 100", where everything is stored,
and how you can ask the data questions later (`datebot explain`, or your own analysis).

## The pipeline

```
photo files ──CLIP image encoder──▶ 512 numbers per photo ─┐
bio / prompts ─CLIP text encoder──▶ 512 numbers per bio   ─┼─▶ one feature vector per profile
                                                           │        │
your likes / passes ───────────────────────────────────────┘        ▼
                                                      logistic regression ─▶ P(like) × 100
```

Everything runs on your machine. The only network access is a one-time download of the CLIP weights.

## What CLIP is

CLIP is a neural network trained by OpenAI-style contrastive learning on hundreds of millions of image/caption
pairs. It has two halves: an **image encoder** and a **text encoder**. Both turn their input into a list of numbers
(an *embedding*, here 512 floats) in the **same space**, trained so that a photo and a caption that describes it end
up close together. "Close" is measured by cosine similarity (the dot product of two unit-length vectors).

Datebot uses the open_clip implementation, model `ViT-B-32`, weights `laion2b_s34b_b79k`
(`EMBED_ID = "ViT-B-32/laion2b_s34b_b79k"` in `embed.py`). Vectors are L2-normalised (length 1).

An embedding is not a list of features like "has glasses = 1". Each of the 512 numbers is a mixture of many visual
concepts; no single number means anything on its own. The meaning is in directions and distances: photos with
similar content have similar vectors.

## Do I need to say what I care about up front?

**No.** You only rate: like or pass. You never tag glasses, gym photos, hair colour or anything else.

This works because the embedding already contains far more than you would tag, and because the text encoder lets you
*ask about* an attribute afterwards. To test "glasses", embed the caption "a photo of a person wearing glasses" and
take its dot product with every stored photo vector. High similarity means CLIP thinks the photo shows glasses. That is
a **zero-shot detector**, built after the fact from a sentence, with no labelled training data.

That is what `datebot explain` does: for each attribute it scores every photo, reduces that to one number per profile
(the best photo, or the average over photos for things like "how many gym photos"), and measures how well it separates
your likes from your passes. You can add any attribute by writing a caption:

```bash
uv run datebot explain                                  # all built-in attributes
uv run datebot explain --group styling --group body     # only some groups
uv run datebot explain --prompt "a photo of a person in a red dress"
uv run datebot explain --examples 3                     # show the photos each detector scored highest
```

So you can think of new things to check next month and run them against ratings you made today. Nothing needs
re-rating or re-embedding.

### Limits of looking things up afterwards

- **A zero-shot detector is only as good as CLIP's understanding of the caption.** It is solid for glasses, gym,
  swimwear, dogs and hair colour; shakier for body shape and weaker still for ethnicity, where it is also known to carry
  biases from its training data. Always check with `--examples`, which prints the photos that scored highest for each
  attribute, so you can see what the detector actually found.
- **Correlation, not causation.** If you like profiles scoring high on "glasses", you might really be liking a style
  that glasses-wearers share. The report says "tends to separate your likes from passes", not "you like glasses".
- **Small data is noisy.** With 14 likes nearly everything is chance. `explain` shows a Benjamini-Hochberg corrected
  q-value so that, across ~38 attributes tested at once, a lucky one does not look like a finding. `*` marks the ones
  that survive. Expect nothing to be flagged until you have a few dozen likes.
- **The trained model does not use the attributes.** It never sees "glasses". It learns one direction in the 512-d
  space from your ratings. `explain` is a separate lens for *describing* that taste, not part of the scoring.
- **If an attribute is a hard rule for you, make it a filter**, not a learned preference (`datebot filters`).

## What the model actually learns from

For each profile (`model.featurize`):

| block | size | meaning |
|---|---|---|
| mean of photo embeddings | 512 | overall look across photos |
| max of photo embeddings (per dimension) | 512 | a standout photo is not averaged away |
| lead (first) photo embedding | 512 | the photo you see first gets its own block, so the model can learn how much it matters versus the rest |
| text embedding, mean over lines | 512 | overall tone of the bio + prompts + interests; zeros if the profile has no text |
| text embedding, max over lines | 512 | one standout prompt is not averaged away |
| has_text flag | 1 | so "no bio" is distinguishable from "bio about nothing" |

Bio, each prompt answer, each interest and each descriptor are embedded **separately** and then pooled, because CLIP
truncates any single text at 77 tokens (a prompt-heavy Hinge profile would otherwise lose most of its prompts).

A `LogisticRegression` (regularised, `class_weight="balanced"`, since you pass on most profiles) is fitted to your
likes/passes. The score is `P(like) × 100`. Only your own ratings (`manual` and `app` origin) are labels: the bot's
own swipes (`auto`) are never used for training.

`datebot eval` tells you whether the model beats chance on profiles it was not trained on.

## Where things are stored

Everything personal is in `data/` (gitignored; override the location with `DATEBOT_HOME`).

| path | contents |
|---|---|
| `data/datebot.db` | SQLite: profiles, photo records, **embeddings**, ratings |
| `data/photos/` | the photo files themselves (the originals the embeddings were computed from) |
| `data/models/*.joblib` | trained models (named snapshots) |
| `data/filters.json` | your hard filters |
| `data/autoswipe.jsonl` | the auto-swipe journal |

### How the embeddings are stored

In `datebot.db`:

- `photos.embedding`: a `BLOB` holding the raw bytes of 512 little-endian `float32` values = **2048 bytes per photo**.
  Next to it, `photos.embed_model` records which model made it (the `EMBED_ID` above).
- `profiles.text_embedding` and `profiles.text_embed_model`: the bio/prompt text, one per profile, stored as the
  pooled `[mean, max]` of its per-line embeddings (1024 floats). `text_embed_model` also carries the layout version
  (`db.TEXT_LAYOUT`), so a layout change re-embeds from the stored text automatically.

They are saved the moment they are computed, by `datebot embed` or automatically by `train`, `score`, `eval`, `explain` and auto-swipe. Capture and import only store profiles; they do not embed. They are
plain columns, so any tool that reads SQLite can get at them. At 2 KB each, even thousands of profiles are a few
megabytes.

### Are they preserved?

Yes. These survive:

- `datebot reset-model`: deletes only `data/models/<name>.joblib`.
- `datebot reset-ratings`: deletes only your rating rows; profiles, photos and embeddings stay.
- Retraining, `eval`, `explain`, capture, auto-swipe: none of them remove embeddings.

Two cases to know about:

1. **Changing the CLIP model overwrites them.** If `EMBED_ID` in `embed.py` is changed (a better CLIP, say), the next
   `datebot embed` re-embeds every photo and replaces the old vector in place; only one embedding per photo is kept.
   The photo files are kept, so nothing is lost permanently, but if you want the old vectors for comparison, copy
   `data/datebot.db` first. Saved models trained on the old embeddings stop matching and must be retrained.
2. **They are only as durable as the file.** Back up `data/` (it is not in git). Deleting `data/photos/` does not
   delete embeddings, but you can no longer re-embed or look at the pictures.

## Analysing the vectors yourself

```python
import sqlite3
import numpy as np

con = sqlite3.connect("data/datebot.db")
rows = con.execute(
    "SELECT id, profile_id, path, embedding FROM photos WHERE embedding IS NOT NULL"
).fetchall()
photo_ids = [r[0] for r in rows]
profile_ids = [r[1] for r in rows]
paths = [r[2] for r in rows]
X = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])   # (n_photos, 512), unit length

# your ratings, latest per profile, excluding the bot's own swipes
liked = dict(con.execute(
    "SELECT profile_id, liked FROM ratings WHERE origin != 'auto' ORDER BY id"
).fetchall())   # later rows win, so this is the latest rating

# "which photos look most like 'glasses'?"  (needs the same CLIP model: use the project's helper)
from datebot import embed
text = embed.embed_texts(["a photo of a person wearing glasses"])[0]
top = np.argsort(-(X @ text))[:10]
print([paths[i] for i in top])
```

Ideas that work well on this data: cluster the vectors (k-means or UMAP) and look at what each cluster of photos has
in common; find the photos nearest to your most-liked profile; or compare the mean vector of liked versus passed
photos and ask CLIP which captions are closest to that difference.

## Resetting, in terms of the above

| to forget… | run | embeddings |
|---|---|---|
| the trained model | `datebot reset-model` | kept |
| your ratings (start teaching again) | `datebot reset-ratings` | kept |
| everything | delete `data/` | gone |
