# The report job. Python 3.11 on arm64 (Graviton: cheaper, and the build script
# downloads aarch64 wheels). The package is a plain zip: pandas + numpy +
# requests + anthropic + the ehf.research.reporting package. No pyarrow,
# no Docker needed to build it.

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "report" {
  function_name = var.name
  description   = "Eagle Hill daily market report: screens -> S3 -> Claude narrative -> SES"
  role          = aws_iam_role.lambda.arn
  handler       = "ehf.research.reporting.lambda_handler.handler"
  runtime       = "python3.11"
  architectures = ["arm64"]
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s

  s3_bucket        = aws_s3_bucket.reports.id
  s3_key           = aws_s3_object.lambda_zip.key
  source_code_hash = filebase64sha256(var.lambda_zip)

  ephemeral_storage {
    size = 1024
  }

  environment {
    variables = {
      EHF_FMP_SECRET_ARN           = aws_secretsmanager_secret.fmp.arn
      EHF_ANTHROPIC_SECRET_ARN     = aws_secretsmanager_secret.anthropic.arn
      EHF_REPORT_SIGNER_SECRET_ARN = aws_secretsmanager_secret.signer.arn
      EHF_REPORT_LINK_DAYS         = tostring(var.link_days)
      EHF_REPORT_BUCKET            = aws_s3_bucket.reports.id
      EHF_REPORT_PREFIX            = "daily-report"
      EHF_REPORT_EMAIL_FROM        = var.report_email
      EHF_REPORT_EMAIL_TO          = join(",", distinct(concat([var.report_email], var.report_recipients)))
      EHF_REPORT_MODEL             = var.narrative_model
      EHF_REPORT_EFFORT            = var.narrative_effort
      EHF_REPORT_MIN_MARKET_CAP    = tostring(var.min_market_cap)
      EHF_REPORT_MIN_AVG_VOLUME    = tostring(var.min_avg_volume)
      EHF_REPORT_FETCH_WORKERS     = tostring(var.fetch_workers)
      PYTHONDONTWRITEBYTECODE      = "1"
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda, aws_iam_role_policy.lambda]
}

# One run at a time: a retry must never overlap a slow run.
resource "aws_lambda_function_event_invoke_config" "report" {
  function_name                = aws_lambda_function.report.function_name
  maximum_retry_attempts       = 1
  maximum_event_age_in_seconds = 3600
}
