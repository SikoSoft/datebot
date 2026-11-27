# Dating-app API endpoints

What datebot knows about the apps' (undocumented, private) APIs: endpoint, purpose, and data contract. None of these
are public or stable; they were learned from our own captured traffic and can change without notice.

**Rules for this file**
- Add every endpoint we discover or rely on, in the app's section, as soon as it is seen. Update it when a parser changes.
- Mark each contract with its status: **Verified** (read from real traffic and parsed successfully), **Seen** (observed
  in traffic, purpose inferred from the path, body not inspected) or **Unverified** (from community docs or a guess).
  Never present a guess as verified; do not invent field names.
- No real data here: no user IDs, tokens, names, URLs with IDs or message text. Use placeholders and field types.
- Datebot only listens to the user's own traffic (see readme warnings). Acting on an account is a separate decision.

Code that implements each contract: `src/datebot/capture/{hinge,badoo,tinder}.py`. Raw dumps for mapping new calls
come from `datebot capture hinge-probe` (phone, mitmproxy) and `capture/discover.py` (Chrome), saved under the
gitignored `data/`.

---

## Hinge (iPhone app, via mitmproxy)

Capture: phone proxied through mitmproxy on this Mac (`datebot capture hinge`). Status of access: **not
certificate-pinned**; Hinge hosts were readable (Nov 2026). Apple/Google/Yahoo background hosts on the phone do refuse
the proxy certificate, which is unrelated.

### Hosts

| Host | Use |
|---|---|
| `prod-api.hingeaws.net` | Main REST API, JSON over HTTP/2. Everything below unless noted. |
| `media.hingenexus.com` | Photo/video CDN. Photo URLs from `content/v2/public` are plain `.jpg` GETs (no auth seen needed by our downloader). Videos arrive as `video/upload/...` with a `.jpeg` poster; voice answers as `.aac` (206 partial). |
| `chat.hingeprod.net`, `presence.hingeprod.net` | gRPC (`chat.v1.ChatService/ListConversations`, `grpc.health.v1.Health/Watch`). Matches/chat; not needed for rating. |
| `client-telemetry.hingeprod.net`, `events.split.io`, `telemetry.split.io`, `*.appsflyersdk.com`, `odf.app-ads-services.com` | Telemetry, feature flags and ads/attribution. Ignore. |

### Feed and profile data (used by datebot)

All **Verified**. Profiles are not in the feed itself: the feed gives IDs, then two calls with the same ID list return
the profile fields and the photos/prompts, which must be joined by `userId` (`HingeSession`).

**`POST /rec/v2`** — the recommendation feed.
- Request: `{newHere: bool, playerId: str, filterCircleMembers: bool, activeToday: bool}`.
- Response 200: `{feeds: [{id: int, origin: str, subjects: [{subjectId: str, pairing: null, ratingToken: str, recId: str, enhancements: null}] (about 20), permission: str, preview: {subjects: [...] (about 6), permission: str}, viewToken: str}], activePills: [{pillType, permission, id, displayOrder}], cacheControl: null}`.
- Returns **304 with an empty body** when the app already holds the current feed (seen after restarts); a body only
  arrives once the feed has changed (e.g. after rating profiles).
- `ratingToken` (per subject) must go back in the rating call; `viewToken` goes back in the profile calls.
- Datebot does not parse this call; the profile calls below carry everything it needs.

**`POST /user/v3/public`** — profile fields.
- Request: `{ids: [userId...] (about 26), viewToken: str}`.
- Response: `[{userId: str, profile: {age: int, firstName: str, lastName: str (empty), height: int (cm), location: {name: str}, works: str, jobTitle: str, educations: [str], hometown: str, selfieVerified: bool, didjustJoin: bool, genderIdentityId: int, sexualOrientations: [int], relationshipTypeIds: [int], ...}}]`. Optional fields (e.g. `jobTitle`, `languagesSpoken`, `religions`, `pronouns`, `ethnicities`) are simply absent when unset.
- Numeric lifestyle fields (`drinking`, `smoking`, `drugs`, `marijuana`, `children`, `familyPlans`, `datingIntention`, ...) are **enum ids whose meaning we have not mapped**; datebot ignores them.
- Used for: name, age, `height` -> `extra.height_cm`, `location.name` -> `extra.city`, `works`/`jobTitle`/`hometown`/`educations` -> `extra.descriptors`.

**`POST /content/v2/public`** — photos and prompt answers.
- Request: same as `/user/v3/public`.
- Response: `[{userId: str, content: {photos: [{url: str, cdnId, contentId, sourceId, source: "camera"|"facebook"|"photoFinder", pHash, width, height, caption: str, selfieVerified: bool, boundingBox: {topLeft, bottomRight}, promptId?: str}] (4-6, display order), answers: [{contentId, position: int, questionId: str, type: "text"|"voice", response: str, transcriptionMetadata: {}}]}}]`.
- Used for: photos in order (`photos.position`), text answers sorted by `position` -> `extra.prompts` as "Question: answer". Voice answers have no text and are skipped.

**`POST /prompts`** — prompt catalogue.
- Response: `{prompts: [{id: str, prompt: str, isSelectable: bool, placeholder: str, isNew: bool, categories: [], contentTypes: []}] (about 450), categories: [{name, slug, isVisible, isNew}]}`.
- Maps an answer's `questionId` to its question text. The app only fetches it at launch, so datebot caches the map in
  `data/hinge-prompts.json`; without it answers are stored without their question.

### Your swipes

**`POST /rate/v2/initiate`** — **Verified** — send a like or pass.
- Request: `{subjectId: str, ratingToken: str (from /rec/v2), rating: str, ratingId: str (uuid), sessionId: str (uuid), origin: str ("compatibles" seen), hasPairing: bool, created: str (timestamp), initiatedWith?: "standard", hcmRunId?: str, content?: {comment: str, prompt: {contentId, question, answer}}}`.
- `rating`: `"note"` = like, sent with a `content` comment on a prompt (**Verified**: the daily like counter dropped on each);
  `"skip"` = pass, no `content` (**Verified**). Any other value (a like without a comment, a rose, ...) has not been
  seen: datebot ignores it and reports it as an "unrecognised rating".
- Response 200: `{limit: {likes: int, superlikes: int, snoozes: int, skipUndos: int}}` for a like (remaining quota); `{limit: null}` for a pass.
- A swipe on a profile datebot never captured is counted but not stored (nothing to rate). Hinge answers `POST /rec/v2` with 304 and shows cached cards, so the addon drops `If-None-Match` / `If-Modified-Since` from `/rec/v2`, `/user/v3/public` and `/content/v2/public` requests to get full responses.
- Only a 200 is recorded as a rating. Order matters: the profile is stored before its swipe.

**`POST /flag/textreview`** — **Verified** — moderation check run on your comment before a like is sent.
- Request: `{receiverId: str, text: str}`. Response: `{hcmRunId: str, isHarmful: bool}`. The `hcmRunId` is echoed in the
  following `/rate/v2/initiate` as `hcmRunId`. Not used by datebot.

### Other calls seen (Seen: purpose inferred from the path, body not inspected)

| Call | Likely purpose |
|---|---|
| `GET /user/v3` | Your own profile fields. |
| `GET /content/v2` | Your own photos and prompts. |
| `GET /like/v2` | Incoming likes (small response, may be a preview). |
| `GET /standouts/v3` | The "Standouts" tab (304 when unchanged). |
| `GET /likelimit` | Remaining likes for the day. |
| `GET /connection/v2` | Matches / connections. |
| `GET /config/v3?country=XX` | App configuration. |
| `GET /boost`, `GET /store/boost`, `POST /store/v6/product`, `POST /offers/v2/eligible`, `POST /v1/personalized` | Boost, store and offers. |
| `GET /profilestate/profile`, `GET /user/circle/progress`, `GET /user/v3/relocationlimit`, `GET /user/export/status` | Account state and settings. |
| `GET /selfieverification`, `GET /selfieverification/eligibility`, `GET /consent/user/blocking` | Verification and consent. |
| `GET /notification/v1/inbox`, `GET /notification/v1/settings/chats`, `GET /inappnotifications/v1/views` | Notifications. |
| `GET /statement/warning`, `GET /statement/removedcontent`, `GET /guidance/v1` (404) | Moderation/guidance messages. |
| `POST /metric/analytics/v1`, `POST /metric/v2` | Analytics (202). Ignore. |

---

## Badoo (web, badoo.com, via Chrome)

Capture: Playwright attached to a normal Chrome (`datebot capture badoo`). **Verified** on real traffic.
Everything is `POST /mwebapi.phtml?<CALL_NAME>` with a JSON envelope `{"body": [{"message_type": N, ...}]}`.

**`SERVER_GET_ENCOUNTERS`** — the card stack. Response body holds `message_type 84` -> `client_encounters.results[].user`:
`{user_id, name, age, albums: [{photos: [{large_url|preview_url}]}], profile_fields: [{id, name, display_value, type}], displayed_about_me: [str], tiw_idea: {tiw_phrase}, city: {name}, interests: [{name}]}`.
- Bio = the `aboutme_text` field. Prompts are `profile_fields` of `type 26` ("question: answer"). Height is the
  `height` field ("178 cm") -> `extra.height_cm`. Labelled fields (relationship, children, smoking, drinking, work,
  education, ...) become descriptors. No distance is exposed, only a city, so the distance filter cannot apply.

**`SERVER_ENCOUNTERS_VOTE`** — your swipe. Request holds `message_type 80` -> `server_encounters_vote: {person_id, vote}`.
- `vote 2` = like, `vote 3` = pass (**Verified**: 6 likes and 10 passes of the user's own session). Other codes are ignored.

Other hosts/calls on the site were not mapped. Use `capture/discover.py` (raw dump) before parsing anything new.

---

## Tinder (web, tinder.com, via Chrome)

Capture: Playwright attached to Chrome (`datebot capture tinder`, `--dump DIR` saves raw responses). Hosts:
`gotinder.com`, `tinder.com`. Tinder has no public API; paths come from community documentation and field names are
partly verified (height, below), the rest best-effort and parsed defensively (missing fields are skipped, never fatal).

**`GET /v2/recs/core`** — the card stack. Response: `{data: {results: [{user: {...}, distance_mi}]}}` (**Unverified** shape
overall; height confirmed).
- Per user: `_id`, `name`, `birth_date` (age is computed), `bio`, `photos: [{url, processedFiles: [{url, width}]}]` (widest
  rendition used), `jobs: [{title: {name}, company: {name}}]`, `schools`, `user_interests`, `city: {name}`,
  `selected_descriptors: [{name, choice_selections, section_name, measurable_selection: {unit_of_measure, value}}]`.
- Height (**Verified** on real traffic): the descriptor with `section_name == "Height"` and `measurable_selection`
  in `cm` -> `extra.height_cm`.
- Cards without `user` (ads, boosts) are skipped.

**`/like/{id}`, `/like/{id}/super`, `/pass/{id}`** — your swipes, recognised from the URL alone (HTTP method not checked)
(`like` = liked, `pass` = not). Super likes count as likes. The first swipe of each session is ignored by default
because Tinder opens with a near-universal "like" that says nothing about taste (`--keep-first` overrides).

---

## Profile links (UI "Open in app")

`web/links.py` builds a link from `source` + stored `external_id`; the UI shows it only when one exists.
- Tinder `https://tinder.com/app/profile/{_id}`: **Unverified**. Captured `recs/core` users (22 fields) carry no username
  or share slug, only `_id`; Tinder's real share links use a separate slug we have not seen. May not open for people you
  have not matched with.
- Badoo `https://badoo.com/profile/{user_id}`: **Unverified** for unmatched people. No URL fields appear in captured traffic.
- Hinge: none. No web profiles or public share URL exist.

---

## Adding a new app or endpoint

1. Map raw traffic first (`capture/discover.py` for web, `capture hinge-probe` for phones). Verify field names from it.
2. Write the parser from facts and test it with synthetic fixtures in the same shape (never commit real profiles).
3. Add the endpoint here with its status, then update `CLAUDE.md` if the architecture notes change.
