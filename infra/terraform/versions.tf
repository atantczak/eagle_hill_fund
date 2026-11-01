terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # State lives next to the research results (same account, same region).
  # Locking uses the S3-native lockfile (Terraform >= 1.10), so no DynamoDB table.
  backend "s3" {
    bucket       = "ehf-results-286016340145"
    key          = "terraform/daily-report.tfstate"
    region       = "us-east-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "ehf-daily-report"
      ManagedBy = "terraform"
      Repo      = "eagle_hill_fund/infra/terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
