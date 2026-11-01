# 06:30 America/New_York, Monday-Friday. Bars for the prior session are final by
# then, pre-market news for after-close earnings is in the feed, and the econ
# calendar for the day is ahead. Monday's report covers Friday.

resource "aws_scheduler_schedule" "daily" {
  name        = var.name
  description = "Daily market report"
  group_name  = "default"
  state       = var.schedule_enabled ? "ENABLED" : "DISABLED"

  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.report.arn
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({ source = "scheduler" })

    retry_policy {
      maximum_retry_attempts       = 2
      maximum_event_age_in_seconds = 3600
    }
  }
}
