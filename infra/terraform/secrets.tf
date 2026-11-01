# Secrets are created here with a placeholder and filled OUT of Terraform
# (python infra/ops.py set-secrets) so real keys never enter the state file.
# The Lambda treats a value starting with "REPLACE" as unset.

resource "aws_secretsmanager_secret" "fmp" {
  name                    = "ehf/fmp"
  description             = "Financial Modeling Prep API key for the daily report"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "fmp" {
  secret_id     = aws_secretsmanager_secret.fmp.id
  secret_string = "REPLACE_ME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "anthropic" {
  name                    = "ehf/anthropic"
  description             = "Anthropic API key for the report narrative (optional)"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "anthropic" {
  secret_id     = aws_secretsmanager_secret.anthropic.id
  secret_string = "REPLACE_ME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}
