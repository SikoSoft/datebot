from __future__ import annotations

import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from datebot import db, ingest
from datebot.interactions import InteractionTracker
from datebot.capture.tinder import CapturedProfile, Swipe

SOURCE = "tinder"
Fetcher = Callable[[str], bytes]


def http_fetch(url: str, timeout: float = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


@dataclass
class Counts:
    seen: int = 0            # profiles observed (including ones already stored)
    new: int = 0             # profiles newly stored
    failed: int = 0          # profiles skipped because no photo could be downloaded
    swipes: int = 0          # your swipes recorded as ratings
    unknown_swipes: int = 0  # swipes on profiles we never captured
    interactions: int = 0    # behaviour interactions stored alongside swipes
    telemetry_failed: int = 0  # swipes whose behaviour telemetry could not be stored (the rating still was)
    ignored: int = 0         # your swipes deliberately not recorded (the session's first-profile freebie)


class Recorder:
    """Stores captured profiles and turns your own swipes into labeled ratings (origin='app')."""

    def __init__(self, fetch: Fetcher = http_fetch, source: str = SOURCE, db_path: str | None = None,
                 skip_first_swipe: bool = False,
                 tracker: InteractionTracker | None = None):
        self.fetch = fetch
        self.tracker = tracker
        self._skip_pending = skip_first_swipe
        self.source = source
        self.db_path = db_path
        self.counts = Counts()

    def _download(self, urls: list[str], tmp: Path) -> list[Path]:
        def one(item: tuple[int, str]) -> Path | None:
            i, url = item
            try:
                suffix = Path(urlparse(url).path).suffix.lower()
                path = tmp / f"{i}{suffix if suffix in ingest.IMAGE_EXTS else '.jpg'}"
                path.write_bytes(self.fetch(url))
                return path
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            return [p for p in pool.map(one, enumerate(urls)) if p]  # map keeps photo order

    def record_profiles(self, profiles: list[CapturedProfile]) -> int:
        """Store profiles not seen before. Returns how many were new."""
        new = 0
        with db.session(self.db_path) as conn:
            for p in profiles:
                self.counts.seen += 1
                if self.tracker:
                    self.tracker.profile_ready(p.external_id)  # arrival time, not after the photo downloads
                if db.find_profile_id(conn, self.source, p.external_id) is not None:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    files = self._download(p.photo_urls, Path(tmp))
                    if not files:
                        self.counts.failed += 1  # not stored, so it is retried if it shows up again
                        continue
                    ingest.add_from_files(
                        conn, files, self.source, external_id=p.external_id,
                        name=p.name, age=p.age, bio=p.bio, extra=p.extra,
                    )
                new += 1
                self.counts.new += 1
        return new

    def take_skip(self, origin: str = "app") -> bool:
        """True once, for your first swipe of the session, if skip_first_swipe was requested.

        Tinder opens every session with a near-universal 'like' that free accounts feel obliged to spend a
        like on; that swipe says nothing about your taste, so it is not recorded as a rating."""
        if origin != "app" or not self._skip_pending:
            return False
        self._skip_pending = False
        self.counts.ignored += 1
        if self.tracker:
            self.tracker.advance()  # the next card still starts now, even though this swipe isn't recorded
        return True

    def record_swipe(self, swipe: Swipe, origin: str = "app", at_ms: float | None = None,
                     events: list[dict] | None = None) -> bool:
        """Record a swipe as a rating (origin 'app' = yours, 'auto' = the bot's).

        With a tracker, also stores the behaviour interaction (dwell, input `events`, `at_ms` = when it happened).
        Returns False if the profile was never captured."""
        with db.session(self.db_path) as conn:
            pid = db.find_profile_id(conn, self.source, swipe.user_id)
            rating_id = None if pid is None else db.add_rating(conn, pid, swipe.liked, origin=origin)
            if self.tracker:
                try:  # telemetry is secondary: it must never cost us the rating
                    if self.tracker.record(conn, external_id=swipe.user_id, profile_id=pid, liked=swipe.liked,
                                           origin=origin, rating_id=rating_id, decided_ms=at_ms, events=events):
                        self.counts.interactions += 1
                except Exception:
                    self.counts.telemetry_failed += 1
            if pid is None:
                self.counts.unknown_swipes += 1
                return False
        self.counts.swipes += 1
        return True
