# ADR-0006 — Write-ups live in S3 behind the site, not in git

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

The first two research write-ups (the VIX explainer and its results report) were
committed as PDFs under `docs/reports/`. Binary artifacts in git bloat
every clone, cannot be diffed, and are awkward to share with someone who does
not have the repository. Andrew wants a place to keep the reports, talk about
them, and point family and friends at a specific one.

## Decision

```
ehf/web/          FastAPI + Jinja: landing page, /library, /daily
infra/aws/site/           Dockerfile, pinned requirements, ops.py, terraform/
.github/workflows/site.yml  build -> ECR -> ECS on push to main (OIDC, no keys)
s3://ehf-market-reports-<acct>/library/<slug>/{meta.json, *.pdf, ...}
```

- **Reports are data, not code.** `scripts/build_*.py` write PDFs to the ignored
  `var/reports/`; `infra/aws/site/ops.py publish-report` uploads a
  PDF plus a `meta.json` (title, date, summary, tags, optional markdown body).
  The site lists whatever is in the prefix. The daily report already lands in
  the same bucket, so it shows up in the same library for free.
- **Login is the load balancer's job.** An ALB `authenticate-cognito` rule
  guards `/library*`, `/daily*`, `/logout` and `/me`; the container only reads
  the identity header the ALB forwards. Invite-only Cognito pool, users declared
  in `terraform.tfvars`. No password code in the app.
- **The container knows nothing about research.** It installs five pinned
  packages and copies only `ehf/web`, so a change to the research stack
  never rebuilds or breaks the site, and the image stays small.
- **Separate Terraform state** (`terraform/site.tfstate`) from the daily-report
  stack; the site reads the report bucket through a data source.
- **Deploys from GitHub Actions through an OIDC role** scoped to one ECR repo
  and one ECS service, from `main` and `dev` only.

## Consequences

- Route 53 becomes authoritative for `eaglehillfund.com` (name servers pasted
  into GoDaddy once); ACM renews the certificate itself.
- Running cost ≈ $27/month, almost all of it the ALB. No NAT gateway: tasks sit
  in public subnets with a public IP and a security group that only admits the ALB.
- Terraform owns the task definition's shape; the workflow registers image-only
  revisions and the service ignores `task_definition` drift. After changing env
  or roles in Terraform, re-run the workflow so the service picks the revision up.
- The backtest browser (interactive results, any-ticker charts) is the next
  thing to add to this app; it has a home, a login and a deploy path already.
