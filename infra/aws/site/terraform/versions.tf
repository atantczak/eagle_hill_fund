terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # Same state bucket as the daily-report stack, its own key: the site can be
  # changed or destroyed without touching the report Lambda.
  backend "s3" {
    bucket       = "ehf-results-286016340145"
    key          = "terraform/site.tfstate"
    region       = "us-east-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "ehf-site"
      ManagedBy = "terraform"
      Repo      = "eagle_hill_fund/infra/aws/site/terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

# The reports bucket belongs to the daily-report stack; the site only reads it.
data "aws_s3_bucket" "reports" {
  bucket = var.reports_bucket != "" ? var.reports_bucket : "ehf-market-reports-${data.aws_caller_identity.current.account_id}"
}

locals {
  account_id     = data.aws_caller_identity.current.account_id
  cognito_domain = "${aws_cognito_user_pool_domain.site.domain}.auth.${data.aws_region.current.region}.amazoncognito.com"
  site_url       = "https://${var.domain}"
}
