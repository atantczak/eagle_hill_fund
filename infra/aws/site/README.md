# infra/aws/site — eaglehillfund.com

The research library behind a login: write-ups published from `var/reports/`
and the daily market report, read straight from the reports bucket; and the lab
(`/lab`), which reads a mirror of the Studio's `var/` from the results bucket.
One container on ECS Fargate behind an application load balancer that enforces
a Cognito sign-in. Terraform state `terraform/site.tfstate` (separate from the
daily-report stack). ADR-0006 (the site) and ADR-0009 (the lab in it) have the
reasoning.

```
infra/aws/site/
  terraform/        VPC (2 public subnets, no NAT) + ALB + ACM + Route 53 + Cognito + ECR + ECS + GitHub OIDC role
  Dockerfile        python:3.11-slim, installs requirements.txt, copies ehf/ (tests excluded by .dockerignore)
  requirements.txt  site packages + the research packages the lab needs, pinned
  ops.py            publish-report / list / unpublish / status / logs / user   (boto3, reads .env)
.github/workflows/site.yml   test -> build -> push to ECR -> roll the ECS service (push to main, or manual)
ehf/web/             the FastAPI app (routes in app.py, S3 access in library.py, templates/)
```

## What gets created

| Resource | Name | Purpose |
|---|---|---|
| Route 53 zone | `eaglehillfund.com` | authoritative DNS; apex + www alias to the ALB; certificate validation records |
| ACM certificate | apex + `*.eaglehillfund.com` | TLS on the ALB, DNS-validated, auto-renewing |
| VPC `10.20.0.0/16` | `ehf-site` | two public subnets, internet gateway, two security groups (ALB from the world; task only from the ALB) |
| ALB + listeners | `ehf-site` | 80 -> 443 redirect; www -> apex; every path requires a Cognito login except `/healthz` |
| Cognito user pool + hosted domain + app client | `ehf-site` | invite-only accounts from `terraform.tfvars`, optional authenticator-app MFA, 7-day ALB sessions |
| ECR repository | `ehf-site` | images tagged by git SHA, last 10 kept, scanned on push |
| ECS cluster / task definition / service | `ehf-site` | 0.5 vCPU, 2 GiB, read-only root filesystem plus a scratch volume at `/var/ehf` for the mirrored `var/` (a one-shot root `init` container chowns it to the app user, since Fargate mounts volumes as root), circuit breaker with rollback |
| IAM roles | `ehf-site-execution`, `ehf-site-task`, `ehf-site-deploy` | pull + logs + the `ehf/fmp` secret; read the reports bucket, read/write `var/*` in the results bucket; GitHub Actions deploys via OIDC from `main`/`dev` |
| S3 gateway endpoint | `ehf-site-s3` | free; the mirror's S3 traffic stays inside AWS |
| CloudWatch log group | `/ecs/ehf-site` | container stdout, 60 days |

## First deploy

```bash
cd infra/aws/site/terraform && terraform init
terraform apply -target=aws_route53_zone.main          # 1. zone only -> prints name_servers
#   GoDaddy -> DNS -> Nameservers -> custom: paste the four values. Wait for propagation.
terraform apply                                        # 2. everything (waits for the certificate to validate)
#   push to main (or run the "site" workflow)          # 3. GitHub builds the image and rolls the service
python infra/aws/site/ops.py status                    # 4. running 1/1, target healthy
```

The Cognito invitation email (temporary password) arrives as soon as step 2 creates
the user; it is valid for 7 days. Sign in at https://eaglehillfund.com/library.

## Everyday

```bash
python scripts/build_vix_results_report.py var/reports/X.pdf var/reports/img
python infra/aws/site/ops.py publish-report --slug x --title "..." --date 2026-11-03 \
    --summary "..." --file var/reports/X.pdf --tags vix --body-md notes.md
python infra/aws/site/ops.py list
python infra/aws/site/ops.py user --email dad@example.com --name "Dad"   # and add to terraform.tfvars
python infra/aws/site/ops.py password --email dad@example.com            # hand out a temporary password
python infra/aws/site/ops.py logs --minutes 30
python infra/aws/site/ops.py ideas                                         # the ideas box, newest first
python infra/aws/site/ops.py ideas --set 20261103-170102-ab12cd --status planned --note "After the QQQ study."
# the thread (ADR-0011): a write-up names its study, its runs and the idea behind it
python infra/aws/site/ops.py publish-report ... --study leverage --run 0123456789abcdef="the headline run" --idea 20261103-214217-9cd512
python infra/aws/site/ops.py link --slug vix-backtest-results --study vix_overlay     # thread an older post
python infra/aws/site/ops.py ideas --set 20261103-214217-9cd512 --status done --link /library/leverage-when-bullish
```

Code change under `ehf/` -> merge to `main` -> the workflow deploys (about 3 minutes).
Terraform change to the task definition (env, roles) -> `terraform apply`, then re-run the
workflow so the service moves to the new revision (the service ignores task-definition drift
so that GitHub-registered revisions are not reverted by Terraform).

## The lab on the site

The container has no `var/`. The Studio copies it up every night
(`python -m ehf.research.jobs.sync_var`, in the launchd table) and the container
mirrors `s3://ehf-results-<account>/var/` into `/var/ehf` on demand: database,
scorecards and Cboe cache at startup, daily bars in the background, a run's
arrays or a new symbol on first touch, a re-check once an hour. The site never
logs playground runs (the "log" box is hidden); the Studio is the writer. Env
on the task: `EHF_LAB=on`, `EHF_VAR_DIR`, `EHF_VAR_S3_BUCKET`, `EHF_VAR_S3_PREFIX`,
`EHF_RESULTS_BACKEND=local`, and `FMP_API_KEY` from the `ehf/fmp` secret.
ADR-0009.

```bash
python -m ehf.research.jobs.sync_var --dry-run     # what the next nightly sync would upload
python -m ehf.research.jobs.sync_var               # push it now (e.g. after a study)
```

## The thread: ideas, write-ups, runs

A write-up's `meta.json` can name the `study` its scorecards carry, the `runs` its
charts are built on and the `ideas` that prompted it. The post page then shows
"The backtests behind this write-up" (with headline numbers when the lab is up) and
"Prompted by"; each run page shows the write-up card; the Studies table links the
write-up; an idea with a `link` shows "Read the result". ADR-0011.

## The playbook and the ideas box

`/playbook` runs the two rules the VIX write-up recommended (surprise count, fear curve
flip) on the latest close, from the same strategy code as the backtests, and says IN,
HALF or OUT with the reasons (ADR-0010). It needs the research platform, so it mounts
wherever the lab does. `/ideas` takes research questions and feature requests from
anyone with a login and stores them as `ideas/<id>.json` in the reports bucket; triage
them with `ops.py ideas` (statuses: new, planned, building, done, declined; the note
shows under the idea on the site).

## Running it locally (with the lab)

The lab (`/lab`: every stored backtest charted, the playground, any-ticker
charts) mounts wherever the research platform imports: straight over `var/`
on your machine, over the S3 mirror in the container. From the repo root:

```bash
set -a && . ./.env && set +a
EHF_LAB=on EHF_RESULTS_BACKEND=local EHF_REPORTS_BUCKET=ehf-market-reports-286016340145 \
  .venv311/bin/uvicorn ehf.web.app:app --port 8080 --reload
open http://localhost:8080/lab
```

`EHF_LAB` is `auto` by default (on when `ehf.research` imports, off otherwise),
`on` to insist, `off` to hide it. The playground reads the price cache under
`var/cache` and the results store under `var/results`; a run on a new ticker
fetches its history from FMP once. Nothing is written unless you tick *log*.
Light/dark follows the system; the toggle in the nav remembers your choice, and
`?theme=light` works on any link. See ADR-0008.
(with the `.env` AWS keys exported). Without the ALB in front there is no login and `/me` is empty.

## Costs

ALB ≈ $17/month, Fargate 0.5 vCPU / 2 GiB ≈ $21, Route 53 $0.50, the mirrored `var/` in S3
≈ $0.02, ECR/logs pennies, Cognito free at this size. No NAT gateway by design. Roughly
$39/month (the lab added about $12; `cpu = 256` in `variables.tf` gives back $7 at the cost
of slower playground runs).
