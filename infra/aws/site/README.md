# infra/aws/site — eaglehillfund.com

The research library behind a login: write-ups published from `ehf/server/var/reports/`
and the daily market report, read straight from the reports bucket. One small
container on ECS Fargate behind an application load balancer that enforces a
Cognito sign-in. Terraform state `terraform/site.tfstate` (separate from the
daily-report stack). ADR-0006 has the reasoning.

```
infra/aws/site/
  terraform/        VPC (2 public subnets, no NAT) + ALB + ACM + Route 53 + Cognito + ECR + ECS + GitHub OIDC role
  Dockerfile        python:3.11-slim, installs requirements.txt, copies only ehf/server/site
  requirements.txt  the five runtime packages, pinned (mirror of the `site` extra in pyproject)
  ops.py            publish-report / list / unpublish / status / logs / user   (boto3, reads .env)
.github/workflows/site.yml   test -> build -> push to ECR -> roll the ECS service (push to main, or manual)
ehf/server/site/             the FastAPI app (routes in app.py, S3 access in library.py, templates/)
```

## What gets created

| Resource | Name | Purpose |
|---|---|---|
| Route 53 zone | `eaglehillfund.com` | authoritative DNS; apex + www alias to the ALB; certificate validation records |
| ACM certificate | apex + `*.eaglehillfund.com` | TLS on the ALB, DNS-validated, auto-renewing |
| VPC `10.20.0.0/16` | `ehf-site` | two public subnets, internet gateway, two security groups (ALB from the world; task only from the ALB) |
| ALB + listeners | `ehf-site` | 80 -> 443 redirect; www -> apex; `/library*`, `/daily*`, `/logout`, `/me` require Cognito; `/` and `/healthz` are public |
| Cognito user pool + hosted domain + app client | `ehf-site` | invite-only accounts from `terraform.tfvars`, optional authenticator-app MFA, 7-day ALB sessions |
| ECR repository | `ehf-site` | images tagged by git SHA, last 10 kept, scanned on push |
| ECS cluster / task definition / service | `ehf-site` | 0.25 vCPU, 512 MiB, read-only root filesystem, circuit breaker with rollback |
| IAM roles | `ehf-site-execution`, `ehf-site-task`, `ehf-site-deploy` | pull + logs; read the reports bucket; GitHub Actions deploys via OIDC from `main`/`dev` |
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
python ehf/server/scripts/build_vix_results_report.py ehf/server/var/reports/X.pdf ehf/server/var/reports/img
python infra/aws/site/ops.py publish-report --slug x --title "..." --date 2026-11-03 \
    --summary "..." --file ehf/server/var/reports/X.pdf --tags vix --body-md notes.md
python infra/aws/site/ops.py list
python infra/aws/site/ops.py user --email dad@example.com --name "Dad"   # and add to terraform.tfvars
python infra/aws/site/ops.py logs --minutes 30
```

Code change under `ehf/server/site/` -> merge to `main` -> the workflow deploys (about 3 minutes).
Terraform change to the task definition (env, roles) -> `terraform apply`, then re-run the
workflow so the service moves to the new revision (the service ignores task-definition drift
so that GitHub-registered revisions are not reverted by Terraform).

Local run: `EHF_REPORTS_BUCKET=ehf-market-reports-286016340145 uvicorn ehf.server.site.app:app --reload`
(with the `.env` AWS keys exported). Without the ALB in front there is no login and `/me` is empty.

## Costs

ALB ≈ $17/month, Fargate 0.25 vCPU ≈ $9, Route 53 $0.50, ECR/logs pennies, Cognito free at
this size. No NAT gateway by design. Roughly $27/month.
