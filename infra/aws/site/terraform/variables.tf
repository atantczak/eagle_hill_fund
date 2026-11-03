variable "region" {
  description = "AWS region (the EHF account's resources live in us-east-1)."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Base name for every resource."
  type        = string
  default     = "ehf-site"
}

variable "domain" {
  description = "Apex domain served by the site. Route 53 becomes authoritative for it (paste the zone's name servers into the registrar)."
  type        = string
  default     = "eaglehillfund.com"
}

variable "reports_bucket" {
  description = "Bucket the daily report writes to and the site reads from. Empty = ehf-market-reports-<account>."
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository (owner/name) allowed to deploy through OIDC."
  type        = string
  default     = "atantczak/eagle_hill_fund"
}

variable "deploy_branches" {
  description = "Branches whose GitHub Actions runs may assume the deploy role."
  type        = list(string)
  default     = ["main", "dev"]
}

variable "users" {
  description = "People who may sign in. Each gets a Cognito account and an invitation email with a temporary password; nobody can self-register."
  type = list(object({
    email = string
    name  = string
  }))
}

variable "protected_paths" {
  description = "Load-balancer path patterns that require a Cognito login (max 5). Everything else is public."
  type        = list(string)
  default     = ["/library*", "/daily*", "/logout", "/me"]
}

variable "session_timeout_s" {
  description = "How long a login lasts before the load balancer asks again (seconds)."
  type        = number
  default     = 604800 # 7 days
}

variable "mfa" {
  description = "Cognito MFA setting: OFF, OPTIONAL (users can enrol an authenticator app) or ON."
  type        = string
  default     = "OPTIONAL"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)."
  type        = number
  default     = 256
}

variable "memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 512
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "log_retention_days" {
  type    = number
  default = 60
}
