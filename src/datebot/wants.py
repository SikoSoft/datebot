"""What the profiles *say* they want and like, for the /wants page.

Unlike `insights.py` (your like rate per attribute) this describes the profiles themselves: how many say a given
thing. It covers every stored profile, rated or not; for the rated ones it also shows your like rate, so you can
see whether a theme goes with the people you like.

Four sections:
  * qualities - what they ask of a partner ("make me laugh", "honest", "tall"). Only *seeking* sentences count
    (looking for / you are / someone who / söker / du är ...), so "I'm funny" is not "wants funny".
  * avoid     - what they say they do not want (games, hookups, ...), from negated or dealbreaker sentences.
  * prompts   - the structured "Question: answer" prompts (dealbreaker, ideal first date, ...), answers tallied.
  * interests - the interest tags, tallied as written.

The phrase lists are English plus the Swedish/Danish/Portuguese this data mostly comes in; they are heuristics,
so each row keeps a few quoted examples to let you check them.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict

from datebot import db, fake, insights

MAX_EXAMPLES = 4
MIN_PROMPT_QUESTION_N = 5
MAX_PROMPT_ANSWERS = 8
MAX_INTERESTS = 40
SNIPPET = 170

# A sentence that is about the partner she wants (rather than about herself).
SEEKING = re.compile(
    r"\b(looking for|look for|seeking|searching|want(s|ing)?|need(s)?|must|should|hoping|wish|ideal|"
    r"someone|somebody|a (man|guy|gentleman|partner)|my (match|partner)|you('| a)?re|you are|you have|you can|you know|"
    r"you like|you love|if you|make me|makes me|show me|treat me|bring|"
    r"s[oö]ker|letar|vill ha|[oö]nskar|dr[oö]mmer|du [aä]r|du ska|du har|du kan|du vill|n[aå]gon som|en (man|kille|snubbe)|"
    r"s[oø]ger|leder efter|du er|en mand|en fyr|"
    r"procuro|busco)\b",
    re.IGNORECASE,
)

QUALITIES: dict[str, str] = {
    "humour / makes me laugh": r"\b(humou?r\w*|funny|make[s]? me laugh|laugh\w*|witty|playful|jokes?|skratta\w*|sk[aä]mt\w*|rolig\w*|kul|"
                               r"sj[oø]v\w*|grine\w*|morsom\w*|engra[cç]ad\w*|bem humorad\w*)\b",
    "honest / open": r"\b(honest(?!ly)\w*|truthful|authentic|genuine|transparent|[aä]rlig\w*|[æa]rlig\w*|sann\w*|sincer\w*|verdadeiro|honesto)\b",
    "loyal / faithful": r"\b(loyal\w*|faithful|trustworth\w*|trust|lojal\w*|trogen|trofast|trovärdig\w*|fiel|leal)\b",
    "kind / caring": r"\b(kind\w*|caring|nice|sweet|warm\w*|gentle\w*|thoughtful|considerate|empath\w*|sn[aä]ll\w*|omt[aä]nksam\w*|"
                     r"varm\w*|sn[æe]ll\w*|omsorgsfuld\w*|carinhos\w*|gentil)\b",
    "good communication": r"\b(communicat\w*|talk(ing)? (about|openly)|conversations?|kommunik\w*|prata|samtal\w*|snakke|samtale\w*|"
                          r"emotionally (available|intelligent|mature))\b",
    "respect / treats me well": r"\b(respect\w*|treat(s)? me|appreciat\w*|respekt\w*|v[aä]rdera\w*|behandl\w*)\b",
    "confident": r"\b(confident|confidence|self[- ]assured|secure in|sj[aä]lvs[aä]ker|sj[aä]lvf[oö]rtroende|selvsikker|selvtillid)\b",
    "takes initiative / a real man": r"\b(initiative|take the lead|takes charge|gentleman|real man|knows how to be a man|"
                                     r"man who knows|manly|ta initiativ\w*|herre|gentleman\w*|rigtig mand|riktig man|stabil man)\b",
    "ambitious / hard-working": r"\b(ambiti\w*|driven|hard[- ]?working|career|goal[- ]oriented|motivated|ambiti[oö]s\w*|m[aå]lmedveten\w*|"
                                r"flitig\w*|arbetsam|ambiti[oø]s\w*|trabalhador\w*)\b",
    "stable / grown-up": r"\b(stable|mature|grown[- ]?up|settled|secure|responsible|stabil\w*|mogen|vuxen|ansvarsfull\w*|stabil\w*|"
                         r"moden|voksen|maduro|est[aá]vel)\b",
    "serious / ready for a relationship": r"\b(serious|committed|commitment|long[- ]term|relationship|marriage|settle down|ready for|"
                                          r"seri[oö]s\w*|f[oö]rh[aå]llande|k[aä]rlek|allvar|sambo|gifta|alvor\w*|forhold|k[aæ]rlighed|s[eé]ri[oa]|relacionamento)\b",
    "adventurous / spontaneous": r"\b(advent\w*|spontan\w*|explor\w*|travel\w*|trips?|new experiences|[aä]ventyr\w*|resa|resor|"
                                 r"oplevelser|rejse\w*|viajar|aventur\w*)\b",
    "active / sporty": r"\b(active|sporty|athletic|fit|gym|workout|training|outdoors?|hik\w+|aktiv\w*|sportig\w*|tr[aä]ning\w*|"
                       r"friluft\w*|motion|træning\w*|esportiv\w*)\b",
    "romantic / affectionate": r"\b(romantic\w*|affection\w*|cuddl\w*|hugs?|kiss\w*|touch|passion\w*|romantisk\w*|kram\w*|mysig\w*|"
                               r"k[aä]rleksfull\w*|hygge\w*|kysse\w*|rom[aâ]ntic\w*|carinho)\b",
    "smart / curious": r"\b(smart|intelligent|clever|curious|educated|well[- ]read|intellectual|klok|kunnig\w*|nyfiken\w*|"
                       r"bildad\w*|dygtig\w*|nysgerrig\w*|inteligente)\b",
    "positive / happy": r"\b(positive|happy|cheerful|optimist\w*|good vibes|joy\w*|glad\w*|positiv\w*|lycklig\w*|gl[aä]dje\w*|"
                        r"glad|munter|feliz|alegre)\b",
    "down to earth / calm": r"\b(down[- ]to[- ]earth|calm|relaxed|easy[- ]going|chill|laid[- ]back|lugn\w*|avslappnad\w*|jordn[aä]ra|"
                            r"rolig man|rolig og|afslappet|tranquilo)\b",
    "family-minded": r"\b(family[- ](oriented|minded|man|values)|familj\w*|familie\w*|wants? (kids|children|a family)|father|dad|"
                     r"pappa|fam[ií]lia)\b",
    "good with food / cooks": r"\b(cook\w*|chef|dinner|restaurant\w*|food|laga mat|matlagning|middag\w*|madlavning|cozinh\w*)\b",
    "dresses well / looks after himself": r"\b(well[- ]dressed|groomed|hygiene|smells good|dress well|well[- ]kept|v[aä]lvårdad\w*|"
                                          r"v[aä]lkl[aä]dd\w*|v[aä]ldoftande|pæn|velplejet)\b",
    "tall": r"\b(tall|taller than|l[aå]ng(a|e)?|h[oö]g(a|re)?|h[oø]j(e)?|alto)\b",
    "non-smoker / sober": r"\b(non[- ]?smok\w*|smoke[- ]free|no smoking|don'?t smoke|sober|drug[- ]free|clean|r[oö]kfri\w*|"
                          r"drogfri\w*|nyktr\w*|ikke ryger|r[oø]gfri\w*|n[aã]o fumante)\b",
}
QUALITY_RES = {k: re.compile(v, re.IGNORECASE) for k, v in QUALITIES.items()}

NEGATION = re.compile(
    r"\b(no|not|don'?t|dont|never|without|swipe left|dealbreaker|deal[- ]breaker|red flags?|avoid|"
    r"inga|inget|ingen|inte|aldrig|utan|ikke|ingen|nej|n[aã]o|sem)\b",
    re.IGNORECASE,
)
AVOID: dict[str, str] = {
    "games / drama": r"\b(games?|drama|mind games|dr[aä]ma|spel|lege|jogos)\b",
    "hookups / casual only": r"\b(hook[- ]?ups?|one[- ]night|ons\b|fwb|casual|friends with benefits|bara sex|sex only|knald\w*|n[aã]o procuro sexo)\b",
    "liars / cheaters": r"\b(liars?|lies|lying|cheat\w*|fake|player|ljug\w*|otrogen\w*|l[oø]gn\w*|mentiras?|mentiroso)\b",
    "ghosting / not replying": r"\b(ghost\w*|dry texters?|not replying|no replies|inte svarar|svarar inte|svarer ikke)\b",
    "smokers / drugs": r"\b(smokers?|smoking|drugs?|drug users?|r[oö]kare|knark|narkotika|stoffer|ryger\w*)\b",
    "married / taken": r"\b(married|taken|attached|boyfriend|girlfriend|gift|upptagen|gift mand|casado|comprometido)\b",
    "rude / disrespectful": r"\b(rude|disrespect\w*|arrogant|toxic|ego|jerks?|oh[oö]vlig\w*|respektl[oö]s\w*|arrogant\w*|grov\w*|mal[- ]educado)\b",
    "one-word / lazy openers": r"\b(one[- ]word|(don'?t|not) (stop|start) (at|with) (hi|hey)|just (say(ing)? )?(hi|hey)|dry (texters?|convers\w*)|boring (texters?|convers\w*))\b",
    "insta / snap / whatsapp moves": r"\b(whats ?app|snap(chat)?|insta(gram)?|telegram|kik)\b",
}
AVOID_RES = {k: re.compile(v, re.IGNORECASE) for k, v in AVOID.items()}

_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+|[;•·|]+| - ")


# Badoo's generated "I'm single, straight, 170 cm, ... I don't like smoking" line lists her attributes: not a statement of wants.
GENERATED = re.compile(r"\bsingle, straight\b", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if len(s.strip()) >= 6 and not GENERATED.search(s)]


def _snippet(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= SNIPPET else s[: SNIPPET - 1].rstrip() + "…"


def _split_prompt(line: str) -> tuple[str, str] | None:
    q, sep, a = line.partition(":")
    q, a = q.strip(), a.strip()
    if sep and 3 <= len(q) <= 90 and a:
        return q.rstrip("?").strip(), a
    return None


def seeking_hits(profile: db.Profile) -> dict[str, str]:
    """Qualities she asks of a partner -> one example sentence each."""
    out: dict[str, str] = {}
    prompts = db._strings(profile.extra.get("prompts"))
    for line in [profile.bio or ""] + prompts:
        pq = _split_prompt(line) if line in prompts else None
        # Answers to partner-directed prompts are about the partner as a whole ("What do you value most in a partner?")
        partner_prompt = bool(pq and re.search(r"partner|looking for|green flag|same page|after", pq[0], re.I))
        for s in sentences(pq[1] if pq else line):
            if partner_prompt or SEEKING.search(s):
                for label, rx in QUALITY_RES.items():
                    if label not in out and rx.search(s):
                        out[label] = _snippet(s)
    return out


def avoid_hits(profile: db.Profile) -> dict[str, str]:
    out: dict[str, str] = {}
    prompts = db._strings(profile.extra.get("prompts"))
    for line in [profile.bio or ""] + prompts:
        pq = _split_prompt(line) if line in prompts else None
        dealbreaker_prompt = bool(pq and re.search(r"dealbreaker|deal-breaker|red flag", pq[0], re.I))
        for s in sentences(pq[1] if pq else line):
            if dealbreaker_prompt or NEGATION.search(s):
                for label, rx in AVOID_RES.items():
                    if label not in out and rx.search(s):
                        out[label] = _snippet(s)
    return out


def _row(label: str, pids: list[int], total: int, liked: dict[int, bool], examples: list[dict]) -> dict:
    rated = [liked[p] for p in pids if p in liked]
    return {
        "label": label, "n": len(pids), "share": len(pids) / total if total else 0.0,
        "rated": len(rated), "likes": sum(rated), "rate": (sum(rated) / len(rated)) if rated else None,
        "examples": examples[:MAX_EXAMPLES],
    }


def build(conn: sqlite3.Connection, source: str | None = None, fake_threshold: int | None = None) -> dict:
    """Everything the page shows. Profiles at or above `fake_threshold` on the fake score are left out (None = keep all)."""
    liked = {r["profile_id"]: bool(r["liked"]) for r in insights._rated_rows(conn)}
    photo_counts = {r["profile_id"]: r["c"] for r in conn.execute("SELECT profile_id, COUNT(*) c FROM photos GROUP BY profile_id")}
    marks = db.fake_marks(conn) if fake_threshold is not None else {}

    profiles: list[db.Profile] = []
    sources: Counter[str] = Counter()
    for row in conn.execute("SELECT id FROM profiles ORDER BY id"):
        p = db.get_profile(conn, row["id"])
        if p is None:
            continue
        sources[p.source] += 1
        if source and p.source != source:
            continue
        if fake_threshold is not None:
            manual = marks.get(p.id)
            if manual is True or (manual is None and fake.score(p, photo_counts.get(p.id, 0), None).score >= fake_threshold):
                continue
        profiles.append(p)

    total = len(profiles)
    with_text = sum(1 for p in profiles if db.profile_text(p))
    quality_pids: dict[str, list[int]] = defaultdict(list)
    quality_ex: dict[str, list[dict]] = defaultdict(list)
    avoid_pids: dict[str, list[int]] = defaultdict(list)
    avoid_ex: dict[str, list[dict]] = defaultdict(list)
    answers: dict[str, Counter[str]] = defaultdict(Counter)
    answer_pids: dict[tuple[str, str], list[int]] = defaultdict(list)
    display: dict[tuple[str, str], str] = {}  # first-seen spelling of each answer
    question_pids: dict[str, list[int]] = defaultdict(list)
    interests: dict[str, list[int]] = defaultdict(list)

    for p in profiles:
        for label, s in seeking_hits(p).items():
            quality_pids[label].append(p.id)
            quality_ex[label].append({"id": p.id, "text": s})
        for label, s in avoid_hits(p).items():
            avoid_pids[label].append(p.id)
            avoid_ex[label].append({"id": p.id, "text": s})
        seen_q: set[str] = set()
        for line in db._strings(p.extra.get("prompts")):
            pq = _split_prompt(line)
            if not pq:
                continue
            q, a = pq
            if q not in seen_q:
                seen_q.add(q)
                question_pids[q].append(p.id)
            key = a.casefold().rstrip(".!")
            answers[q][key] += 1
            display.setdefault((q, key), a)
            answer_pids[(q, key)].append(p.id)
        for tag in {t.strip() for t in db._strings(p.extra.get("interests")) if t.strip()}:
            interests[tag].append(p.id)

    def rows(pids: dict[str, list[int]], ex: dict[str, list[dict]]) -> list[dict]:
        out = [_row(k, v, total, liked, ex[k]) for k, v in pids.items()]
        return sorted(out, key=lambda r: (-r["n"], r["label"]))

    prompt_cards = []
    for q, pids in sorted(question_pids.items(), key=lambda kv: -len(kv[1])):
        if len(pids) < MIN_PROMPT_QUESTION_N:
            continue
        ans = []
        for key, c in answers[q].most_common(MAX_PROMPT_ANSWERS):
            ans.append(_row(display[(q, key)], answer_pids[(q, key)], len(pids), liked, []))
        prompt_cards.append({"question": q, "n": len(pids), "answers": ans})

    interest_rows = [_row(k, v, total, liked, []) for k, v in interests.items() if len(v) >= 2]
    interest_rows.sort(key=lambda r: (-r["n"], r["label"]))

    return {
        "summary": {
            "profiles": total, "with_text": with_text, "rated": sum(1 for p in profiles if p.id in liked),
            "likes": sum(1 for p in profiles if liked.get(p.id)), "sources": dict(sources), "source": source,
            "fake_threshold": fake_threshold,
        },
        "qualities": rows(quality_pids, quality_ex),
        "avoid": rows(avoid_pids, avoid_ex),
        "prompts": prompt_cards,
        "interests": interest_rows[:MAX_INTERESTS],
    }
