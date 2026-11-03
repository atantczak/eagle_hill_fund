# infra — where EHF runs

Two execution targets, one repo, no duplication: the Python package is the same
everywhere; `infra/` only decides *where* and *when* a module runs.

```
infra/
  aws/        what runs in AWS — the daily market report
    terraform/      Lambda + EventBridge Scheduler + S3 + Secrets Manager + SES + SNS alarm
    build_lambda.sh builds infra/aws/build/lambda.zip (manylinux aarch64 wheels + ehf subset)
    ops.py          set-secrets / rotate-signer-key / invoke / logs / latest  (boto3, reads .env)
    site/           eaglehillfund.com — ECS Fargate + ALB + Cognito + Route 53 (own Terraform state)
      terraform/    the stack;  Dockerfile + requirements.txt  the image;  ops.py  publish-report / status / logs
  studio/     what runs on the Mac Studio — research jobs on the 28-core / 96 GB box
    jobs.py         THE job table -> launchd agents: render / install / uninstall / status / run
    launchd/        rendered plists (git-ignored; regenerated from jobs.py)
```

| Target | Why there | Scheduler | Logs |
|---|---|---|---|
| **AWS** (`aws/`) | small, must run even if the Studio is off, emails you | EventBridge Scheduler (cron, America/New_York) | CloudWatch `/aws/lambda/ehf-daily-report` |
| **AWS site** (`aws/site/`) | the research library at eaglehillfund.com, behind a login | GitHub Actions deploys on push to `main` | CloudWatch `/ecs/ehf-site` |
| **Studio** (`studio/`) | CPU-heavy research: sweeps, walk-forward, full-universe refresh | launchd user agents (`com.eaglehill.*`) | `var/log/<job>.log` + `<job>.launchd.log` |

Rules of thumb:

* Anything that needs the cores or the local cache runs on the Studio.
* Anything that must be *reliable at 6:30 am regardless of the Studio* runs in AWS.
* A job is a module under `ehf/research/jobs/`; `infra/` never contains logic.
* Secrets come from `.env` locally and Secrets Manager in AWS — never from infra files.

See [`aws/README.md`](aws/README.md) and [`studio/README.md`](studio/README.md).
