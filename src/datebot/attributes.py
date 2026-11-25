"""Descriptive visual attributes per profile (hair colour, body type, glasses, ...) from CLIP zero-shot.

Unlike `explain.py` (which ranks attributes by how well they separate likes from passes), this assigns each
profile a *label* per attribute so the insights page can show like rates per value. It reuses the stored photo
embeddings: only the short prompt texts are embedded here (once, cached on disk), photos are never re-read.

Everything is a noisy guess from a small model. Categories are argmax of a softmax over the prompts of one group,
averaged over the photos where the person is actually visible. Treat the results as directional, and
`datebot explain --examples` is the way to eyeball what the detector really finds.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Callable

import numpy as np

from datebot import config, db

LOGIT_SCALE = 100.0  # CLIP's learned temperature; makes the softmax over a few prompts usefully peaked
UNCLEAR = "unclear"
BASELINE_PROMPT = "a photo"  # what a flag's prompt is compared against
ANY_Z = 2.0  # a flag with agg="any" needs one photo this many standard deviations above the typical photo
MOST_Z = 0.5  # a flag with agg="most" needs this average z over the usable photos
CENTER = 0.5  # how much of each label's average logit (its prompt's bias across your photos) is subtracted
GATE_THRESHOLD = 0.5  # a photo only counts for a group if its gate says the needed thing is visible

# Gates: is the thing the group needs actually in the photo? (prompts for it, prompts against it)
GATES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "person": (
        ("a photo of a woman", "a portrait of a person", "a selfie of a person"),
        ("a photo of a landscape with no people", "a photo of food", "a photo of an animal", "a photo of an object"),
    ),
    "body": (
        ("a full-body photo of a woman standing", "a photo of a woman from the waist up", "a photo of a woman's torso"),
        ("a close-up photo of a face", "a photo of a landscape with no people", "a photo of food or an object"),
    ),
}


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    section: str
    labels: dict[str, tuple[str, ...]]  # label -> prompts describing it
    kind: str = "one_of"  # "one_of": one label per profile | "flag": labels are "yes"/"no"
    gate: str | None = None
    agg: str = "any"  # flags only: "any" (one photo shows it) | "most" (typical of the profile)
    min_conf: float = 0.4  # one_of: below this the profile is labelled "unclear"
    note: str = ""


def _one(key, title, section, labels, **kw) -> Group:
    return Group(key, title, section, {k: tuple(v) for k, v in labels.items()}, **kw)


def _flag(key, title, section, yes: str, **kw) -> Group:
    # Only a "yes" prompt: CLIP does not understand negation ("no glasses" matches glasses), so a flag is scored
    # as how far a photo stands out on the prompt compared with a neutral photo, not as yes-versus-no.
    return Group(key, title, section, {"yes": (yes,)}, kind="flag", **kw)


NOISY = "Rough guess by a small zero-shot model: directional only."

GROUPS: list[Group] = [
    # --- looks (categorical) ---
    _one("hair_colour", "Hair colour", "Looks", {
        "blonde": ["a photo of a woman with blonde hair"],
        "brown": ["a photo of a woman with brown hair", "a photo of a brunette woman"],
        "black": ["a photo of a woman with black hair"],
        "red": ["a photo of a woman with red ginger hair"],
        "grey/white": ["a photo of a woman with grey or white hair"],
        "dyed": ["a photo of a woman with pink, blue or purple dyed hair"],
    }, gate="person", note=NOISY),
    _one("hair_length", "Hair length", "Looks", {
        "short": ["a photo of a woman with short hair", "a photo of a woman with a pixie cut"],
        "medium": ["a photo of a woman with shoulder-length hair"],
        "long": ["a photo of a woman with long hair down past her shoulders"],
        "up": ["a photo of a woman with her hair tied up in a bun or ponytail"],
    }, gate="person", note=NOISY),
    _one("hair_texture", "Hair texture", "Looks", {
        "straight": ["a photo of a woman with straight hair"],
        "wavy": ["a photo of a woman with wavy hair"],
        "curly": ["a photo of a woman with curly hair"],
    }, gate="person", note=NOISY),
    _one("skin_tone", "Skin tone", "Looks", {
        "fair": ["a photo of a woman with very fair pale skin"],
        "light": ["a photo of a woman with light skin"],
        "medium/tan": ["a photo of a woman with olive or tanned skin"],
        "dark": ["a photo of a woman with dark skin"],
    }, gate="person", note=NOISY),
    _one("ethnicity", "Apparent ethnicity (broad)", "Looks", {
        "white / European": ["a photo of a White European woman"],
        "Black": ["a photo of a Black woman"],
        "East Asian": ["a photo of an East Asian woman"],
        "Southeast Asian": ["a photo of a Southeast Asian woman"],
        "South Asian": ["a photo of a South Asian woman"],
        "Middle Eastern / N. African": ["a photo of a Middle Eastern woman"],
        "Latina": ["a photo of a Latina woman"],
    }, gate="person", note="A guess from appearance only, by a model that is least reliable here. Look at examples before trusting it."),
    _one("makeup", "Makeup", "Looks", {
        "none / natural": ["a photo of a woman with no makeup, natural face"],
        "light": ["a photo of a woman with light makeup"],
        "heavy": ["a photo of a woman with heavy makeup, bold lipstick and eyeliner"],
    }, gate="person", note=NOISY),
    _one("expression", "Expression", "Looks", {
        "big smile": ["a photo of a woman smiling broadly showing her teeth"],
        "soft smile": ["a photo of a woman with a slight smile"],
        "serious": ["a photo of a woman with a serious neutral expression"],
        "moody / sultry": ["a photo of a woman with a sultry pouty look"],
    }, gate="person", note=NOISY),
    # --- body (categorical, only where the body shows) ---
    _one("body_type", "Body type", "Body", {
        "petite / slim": ["a photo of a slim petite woman"],
        "athletic": ["a photo of a fit athletic toned woman"],
        "average": ["a photo of a woman with an average build"],
        "curvy": ["a photo of a curvy woman with wide hips"],
        "plus size": ["a photo of a plus-size woman"],
    }, gate="body", note="Only photos that show the body are used. " + NOISY),
    _one("breast_size", "Breast size", "Body", {
        "small": ["a photo of a woman with small breasts, flat chest"],
        "medium": ["a photo of a woman with medium sized breasts"],
        "large": ["a photo of a woman with large breasts"],
        "not visible": ["a photo of a woman wearing a loose baggy sweater, chest not visible"],
    }, gate="body", min_conf=0.45,
        note="Only photos that show the torso are used. This is the weakest detector here: treat it as a hint at most."),
    _flag("cleavage", "Cleavage / low-cut top", "Body", "a photo of a woman wearing a low-cut top showing cleavage", gate="body", note=NOISY),
    _flag("swimwear", "Swimwear / bikini photo", "Body", "a photo of a woman in a bikini or swimsuit", gate="body"),
    _flag("tall_look", "Looks tall", "Body", "a photo of a very tall woman",
          gate="body", agg="most", note="Looks tall in photos, not their stated height."),
    # --- accessories & details (flags) ---
    _flag("glasses", "Glasses", "Details", "a photo of a woman wearing prescription glasses", gate="person"),
    _flag("sunglasses", "Sunglasses", "Details", "a photo of a woman wearing sunglasses", gate="person"),
    _flag("tattoos", "Visible tattoos", "Details", "a photo of a woman with visible tattoos on her skin", gate="person", note=NOISY),
    _flag("piercings", "Visible piercings", "Details", "a photo of a woman with a nose ring or facial piercings", gate="person"),
    _flag("freckles", "Freckles", "Details", "a photo of a woman with freckles", gate="person"),
    _flag("bangs", "Bangs / fringe", "Details", "a photo of a woman with bangs fringe across her forehead", gate="person"),
    _flag("hat", "Hat / cap", "Details", "a photo of a woman wearing a hat or cap", gate="person"),
    # --- style (flags, typical of the profile) ---
    _one("style", "Dominant style", "Style", {
        "casual": ["a photo of a woman in casual clothes, t-shirt and jeans"],
        "sporty": ["a photo of a woman in sportswear"],
        "elegant": ["a photo of a woman in an elegant dress"],
        "alternative": ["a photo of a woman with an alternative goth punk style"],
        "bohemian": ["a photo of a woman in a bohemian or vintage outfit"],
        "sexy / going out": ["a photo of a woman in a tight dress for a night out"],
    }, gate="body", note=NOISY),
    # --- photo content (flags) ---
    _flag("selfies", "Mostly selfies", "Photos", "a close-up selfie taken by the person themselves", agg="most"),
    _flag("mirror_selfie", "Mirror selfie", "Photos", "a mirror selfie holding a phone"),
    _flag("full_body", "Full-body shot", "Photos", "a full-body photo of a woman standing"),
    _flag("group_photo", "Group photo", "Photos", "a group photo of several friends together"),
    _flag("outdoors", "Outdoors / nature", "Photos", "a photo of a person outdoors in nature, hiking, forest or mountains", agg="most"),
    _flag("travel", "Travel photo", "Photos", "a tourist photo at a famous landmark abroad", agg="most"),
    _flag("beach", "Beach / water", "Photos", "a photo of a person at the beach or by the sea"),
    _flag("gym", "Gym / fitness", "Photos", "a photo of a person working out at the gym"),
    _flag("party", "Party / nightlife", "Photos", "a photo of a person at a party or nightclub"),
    _flag("alcohol", "Drinking alcohol", "Photos", "a photo of a person holding a glass of wine or a cocktail"),
    _flag("dog", "With a dog", "Photos", "a photo of a person with a dog"),
    _flag("cat", "With a cat", "Photos", "a photo of a person with a cat"),
    _flag("with_child", "With a child", "Photos", "a photo of a woman with a small child"),
    _flag("with_man", "With a man", "Photos", "a photo of a woman with a man beside her"),
    _flag("car", "Car / vehicle", "Photos", "a photo of a person in a car or on a motorbike"),
    _flag("sports", "Sports / active", "Photos", "a photo of a person skiing, surfing, running or playing sports"),
    _flag("music", "Music / instrument", "Photos", "a photo of a person playing a musical instrument or at a concert"),
    _flag("food", "Food / cooking", "Photos", "a photo of a person cooking or eating food"),
    _flag("formal", "Formal / dressed-up event", "Photos", "a photo of a person at a wedding or formal event"),
    _flag("professional", "Professional-looking photo", "Photos", "a professional studio portrait photograph", agg="most"),
    _flag("filtered", "Heavy filter / edited", "Photos", "a photo with a heavy beauty filter or snapchat effect"),
    _flag("bw", "Black and white", "Photos", "a black and white photo"),
    _flag("face_hidden", "Face hidden / obscured", "Photos", "a photo where the person's face is hidden or turned away"),
]

GROUP_BY_KEY = {g.key: g for g in GROUPS}


def all_prompts() -> list[str]:
    seen: dict[str, None] = {}
    for g in GROUPS:
        for prompts in g.labels.values():
            seen.update(dict.fromkeys(prompts))
    for pos, neg in GATES.values():
        seen.update(dict.fromkeys(pos + neg))
    seen[BASELINE_PROMPT] = None
    return list(seen)


def prompt_embeddings(
    embed_texts: Callable[[list[str]], np.ndarray], embed_id: str, cache_dir=None
) -> dict[str, np.ndarray]:
    """Embed every prompt (cached on disk, keyed by the prompts and the CLIP model)."""
    prompts = all_prompts()
    key = hashlib.sha1("\n".join([embed_id, *prompts]).encode()).hexdigest()[:16]
    cache_dir = cache_dir or config.data_dir() / "cache"
    path = cache_dir / f"attribute-prompts-{key}.npy"
    if path.exists():
        vecs = np.load(path)
    else:
        vecs = embed_texts(prompts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return dict(zip(prompts, vecs))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _label_matrix(prompts_by_label: dict[str, tuple[str, ...]], pe: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([_unit(np.mean([pe[p] for p in ps], axis=0)) for ps in prompts_by_label.values()])


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gate_scores(photo_emb: np.ndarray, pe: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per gate, P(the needed thing is visible) for every photo."""
    out = {}
    for name, (pos, neg) in GATES.items():
        mat = _label_matrix({"pos": pos, "neg": neg}, pe)
        out[name] = _softmax(LOGIT_SCALE * photo_emb @ mat.T)[:, 0]
    return out


def classify(
    profile_ids: list[int], photo_emb: np.ndarray, pe: dict[str, np.ndarray], groups: list[Group] | None = None
) -> dict[int, dict[str, str | None]]:
    """{profile id: {group key: label}}. A label is None when no photo of the profile could show it.

    Categories: softmax over the group's labels (logits partly centred, see CENTER), averaged over the profile's
    usable photos. Flags: z-score of "yes prompt minus neutral baseline" over all usable photos in your data, so
    "yes" means this profile's photos stand out on it relative to the others you have seen."""
    groups = GROUPS if groups is None else groups
    pids = np.asarray(profile_ids)
    gates = gate_scores(photo_emb, pe)
    baseline = photo_emb @ _unit(pe[BASELINE_PROMPT])
    index = {pid: np.flatnonzero(pids == pid) for pid in dict.fromkeys(profile_ids)}
    result: dict[int, dict[str, str | None]] = {pid: {} for pid in index}
    for g in groups:
        usable = gates[g.gate] >= GATE_THRESHOLD if g.gate else np.ones(len(pids), dtype=bool)
        logits = LOGIT_SCALE * photo_emb @ _label_matrix(g.labels, pe).T  # (photos, labels)
        names = list(g.labels)
        if g.kind == "flag":
            score = logits[:, 0] - LOGIT_SCALE * baseline
            ref = score[usable] if usable.any() else score
            score = (score - ref.mean()) / (ref.std() + 1e-9)
        else:
            ref = logits[usable] if usable.any() else logits
            probs = _softmax(logits - CENTER * ref.mean(axis=0))
        for pid, rows in index.items():
            rows = rows[usable[rows]]
            if len(rows) == 0:
                result[pid][g.key] = None
            elif g.kind == "flag":
                hit = score[rows].max() >= ANY_Z if g.agg == "any" else score[rows].mean() >= MOST_Z
                result[pid][g.key] = "yes" if hit else "no"
            else:
                mean = probs[rows].mean(axis=0)
                top = int(mean.argmax())
                result[pid][g.key] = names[top] if mean[top] >= g.min_conf else UNCLEAR
    return result


def profile_attributes(conn: sqlite3.Connection, embed_id: str, embed_texts: Callable | None = None):
    """Labels for every profile that has embedded photos."""
    pids, _, emb = db.all_photo_embeddings(conn, embed_id)
    if not pids:
        return {}
    if embed_texts is None:
        from datebot import embed

        embed_texts = embed.embed_texts
    return classify(pids, emb, prompt_embeddings(embed_texts, embed_id))
