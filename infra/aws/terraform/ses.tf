# SES is in sandbox mode on this account: sender AND recipient must be verified.
# Using one address for both means one verification click. Terraform creates the
# identity; AWS emails a verification link to report_email on first apply.

resource "aws_sesv2_email_identity" "report" {
  email_identity = var.report_email
}

# Extra recipients (SES sandbox: every recipient must be a verified identity).
resource "aws_sesv2_email_identity" "recipients" {
  for_each       = toset(var.report_recipients)
  email_identity = each.value
}
