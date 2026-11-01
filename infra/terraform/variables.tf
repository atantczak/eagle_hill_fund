variable "region" {
  description = "AWS region (the EHF account's resources live in us-east-1)."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Base name for every resource."
  type        = string
  default     = "ehf-daily-report"
}

variable "report_email" {
  description = "Address the report is sent FROM and TO. Must be verified in SES (Terraform creates the identity; you click the link)."
  type        = string
}

variable "alarm_email" {
  description = "Where Lambda failure alarms go (defaults to report_email)."
  type        = string
  default     = null
}

variable "schedule_expression" {
  description = "EventBridge Scheduler cron, evaluated in schedule_timezone. Default: 06:30 Mon-Fri."
  type        = string
  default     = "cron(30 6 ? * MON-FRI *)"
}

variable "schedule_timezone" {
  type    = string
  default = "America/New_York"
}

variable "schedule_enabled" {
  type    = bool
  default = true
}

variable "lambda_zip" {
  description = "Path to the built Lambda package (produced by infra/build_lambda.sh)."
  type        = string
  default     = "../build/lambda.zip"
}

variable "lambda_memory_mb" {
  type    = number
  default = 3008
}

variable "lambda_timeout_s" {
  type    = number
  default = 900
}

variable "narrative_model" {
  description = "Claude model used for the narrative."
  type        = string
  default     = "claude-opus-5-5"
}

variable "narrative_effort" {
  type    = string
  default = "high"
}

variable "min_market_cap" {
  type    = number
  default = 2000000000
}

variable "min_avg_volume" {
  type    = number
  default = 1000000
}

variable "fetch_workers" {
  type    = number
  default = 16
}

variable "log_retention_days" {
  type    = number
  default = 60
}
