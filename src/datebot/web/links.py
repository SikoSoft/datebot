"""Link from a stored profile back to the person's page in their app, where one exists.

Templates are built from the stored external id only and are not confirmed from captured traffic
(see docs/api-endpoints.md, "Profile links"). Tinder has none: tinder.com/app/profile/{id} shows an error page, and the
web app has no share button. Hinge has no web profile at all.
"""
from __future__ import annotations

PROFILE_URLS = {
    "badoo": "https://badoo.com/profile/{id}",  # Unverified for people you have not matched with
}


def profile_url(source: str, external_id: str | None) -> str | None:
    template = PROFILE_URLS.get(source)
    return template.format(id=external_id) if template and external_id else None
