# infra/aws — the daily report on AWS

Everything the daily market report needs in AWS, as Terraform, built and deployed
from this laptop with no Docker. Account `286016340145`, region `us-east-1`.

The website (`site/`) is a second, independent stack in the same account: see
[`site/README.md`](site/README.md).

```
infra/aws/
  terraform/        Lambda + EventBridge Scheduler + S3 + Secrets Manager + SES + SNS alarm
  build_lambda.sh   builds infra/aws/build/lambda.zip (manylinux aarch64 wheels + ehf subset)
  ops.py            set-secrets / rotate-signer-key / invoke / logs / latest   (boto3, reads .env)
```

## What gets created

| Resource | Name | Purpose |
|---|---|---|
| Lambda (py3.11, arm64, 3 GB, 15 min) | `ehf-daily-report` | runs `ehf.research.reporting.lambda_handler` |
| EventBridge Scheduler | `ehf-daily-report` | `cron(30 6 ? * MON-FRI *)` America/New_York |
| S3 bucket | `ehf-market-reports-<acct>` | `daily-report/YYYY/MM/DD/*` + `daily-report/latest/*` + the Lambda zips |
| Secrets Manager | `ehf/fmp`, `ehf/anthropic`, `ehf/report-signer` | API keys and the signer's access key; values set by `ops.py`, never by Terraform (`ehf/fmp` is also read by the site task for the lab) |
| IAM user | `ehf-daily-report-signer` | read-only on the reports prefix; its key presigns the 7-day links in the email |
| SES identity | your address | sender = recipient (SES sandbox) |
| SNS topic + CloudWatch alarm | `ehf-daily-report-alarms` | emails you if the Lambda errors |
| IAM roles | `…-lambda`, `…-scheduler` | least privilege: this bucket, these secrets, SES from this address |

Terraform state: `s3://ehf-results-286016340145/terraform/daily-report.tfstate` (S3 lockfile, no DynamoDB).

## Deploy / update

```bash
infra/aws/build_lambda.sh                                   # 1. build the zip (~1 min)
cd infra/aws/terraform && terraform init && terraform apply # 2. create / update everything
python infra/aws/ops.py set-secrets                         # 3. push keys from .env (FMP_API_KEY, ANTHROPIC_API_KEY)
python infra/aws/ops.py rotate-signer-key                   #    once: access key for the link-signer user
python infra/aws/ops.py invoke --dry-run --max-symbols 100  # 4. smoke test (no S3, no email)
python infra/aws/ops.py invoke                              # 5. real run
```

First apply only: click the SES verification link and confirm the SNS subscription
(both arrive by email). Until the SES link is clicked the Lambda still runs and
stores to S3, but `send_email` fails and the run reports an error.

Code change → rerun steps 1-2 (`source_code_hash` redeploys only when the zip changes).

## What a run produces

`report.html` (self-contained dark page, opens on anything), `report.pdf` (reportlab,
letter), `report.md`, `hits.json`, `market.json`, `universe_table.csv.gz`, `meta.json`,
`narrative.md`. All land under `daily-report/YYYY/MM/DD/` and `daily-report/latest/`.
The email is a short summary with the PDF attached and two presigned links (HTML, PDF)
valid for `link_days` (7, the S3 maximum). Links are signed by the `…-signer` user's
long-lived key because a URL signed with the Lambda's temporary role credentials dies
with that session; if the signer key is missing the links are still generated but
short-lived, and the attached PDF always works.

## Credentials

Terraform and `ops.py` use the standard AWS credential chain. The repo `.env`
currently holds **root** access keys; that works, but create an IAM user/role
with admin-ish rights for Terraform and rotate the root keys off the laptop.
The Lambda itself never sees those keys: it runs under its own least-privilege role.

## Costs

Lambda ~20 s/day at 3 GB ≈ pennies/month. Secrets Manager $0.80/month. S3 and SES
negligible. The narrative is the only meaningful cost: one Claude call per day.
