output "lambda_function" {
  value = aws_lambda_function.report.function_name
}

output "reports_bucket" {
  value = aws_s3_bucket.reports.id
}

output "schedule" {
  value = "${aws_scheduler_schedule.daily.schedule_expression} ${aws_scheduler_schedule.daily.schedule_expression_timezone} (${aws_scheduler_schedule.daily.state})"
}

output "fmp_secret_arn" {
  value = aws_secretsmanager_secret.fmp.arn
}

output "anthropic_secret_arn" {
  value = aws_secretsmanager_secret.anthropic.arn
}

output "next_steps" {
  value = <<-EOT
    1. Check ${var.report_email} for the SES verification email and click the link.
    2. Confirm the SNS alarm subscription email too.
    3. python infra/ops.py set-secrets            # pushes FMP_API_KEY (+ ANTHROPIC_API_KEY if set) from .env
    4. python infra/ops.py invoke --dry-run --max-symbols 100   # smoke test
    5. python infra/ops.py invoke                 # real run: S3 + email
  EOT
}
