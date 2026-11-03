# ADR-0010 — The playbook page and the ideas box

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

The VIX study produced two rules worth acting on, but acting on them meant
opening a notebook. Andrew wants the vetted rules on the site, readable every
day: what the VIX is at, what each rule says, in or out. He also wants Wes, his
dad and himself to be able to propose research questions and site features
somewhere he can review when planning with Claude.

## Decision

**Playbook (`/playbook`, `ehf/web/playbook.py`).** The rules the results
write-up recommended, hard-coded as a short list (`RULES`): the surprise count
(rule 6, the dial) and the fear curve flip (rule 1, the sanity check), with the
settings the study used. The page runs the *same strategy classes* on the
cached SPY / VIX / VIX3M bars, after re-fetching the last ten days from the
vendor (`FMPProvider.refresh_tail`) so the reading is as of the latest close
(or the partial bar during the session). It shows the combined answer
(both in: IN; both out: OUT, the strongest signal; otherwise HALF, straight
from the write-up's recommendation), each rule's reading against its two
thresholds, days in the current state, the recent switches, a chart with the
out-of-market days shaded, and the rule's walk-forward record from its latest
scorecard. The payload is cached for 15 minutes. Adding a rule is adding an
entry to `RULES`; nothing else changes.

**Ideas (`/ideas`, `ehf/web/ideas.py`).** A plain HTML form (no JavaScript,
no `python-multipart`: the body is parsed with `urllib`) that writes one JSON
object per idea to `ideas/<id>.json` in the reports bucket. The author comes
from the Cognito login. Statuses (`new`, `planned`, `building`, `done`,
`declined`) and a note from Andrew are set from the Studio with
`python infra/aws/site/ops.py ideas --set <id> --status planned --note "..."`;
`ops.py ideas` prints the queue for planning sessions. Cross-site posts are
refused via `Sec-Fetch-Site`. The task role may `PutObject` under `ideas/`.

## Consequences

- The playbook needs the research platform, so it mounts with the lab (locally
  and in the container via the S3 mirror); the ideas box mounts everywhere.
- Playbook pages during the session show a partial last bar; the "as of" date
  says so implicitly. The 200-day line is shown but is not part of the dial
  until the combination study is done.
- Ideas live in S3 next to the reports; there is no edit or delete from the
  site, by design. Deleting one is an S3 delete (versioned, 30 days).
