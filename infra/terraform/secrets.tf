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

# {"access_key_id": "...", "secret_access_key": "..."} for the link-signer user.
resource "aws_secretsmanager_secret" "signer" {
  name                    = "ehf/report-signer"
  description             = "Access key of the read-only IAM user that presigns report links"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "signer" {
  secret_id     = aws_secretsmanager_secret.signer.id
  secret_string = jsonencode({ access_key_id = "REPLACE_ME", secret_access_key = "REPLACE_ME" })
  lifecycle {
    ignore_changes = [secret_string]
  }
}
