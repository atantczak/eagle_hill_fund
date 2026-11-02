# Every run's artifacts: report.md / report.html / hits.json / market.json /
# universe_table.csv.gz / meta.json under <prefix>/YYYY/MM/DD/ plus <prefix>/latest/.
# hits.json is the dataset evaluate.py scores later; keep it forever.

resource "aws_s3_bucket" "reports" {
  bucket = "ehf-market-reports-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "reports" {
  bucket                  = aws_s3_bucket.reports.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "reports" {
  bucket = aws_s3_bucket.reports.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "reports" {
  bucket = aws_s3_bucket.reports.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "reports" {
  bucket = aws_s3_bucket.reports.id

  # `latest/` is overwritten daily; don't keep a year of superseded versions of it.
  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Lambda deployment packages (the zip is too big to inline reliably).
resource "aws_s3_object" "lambda_zip" {
  bucket = aws_s3_bucket.reports.id
  key    = "lambda/${var.name}-${filemd5(var.lambda_zip)}.zip"
  source = var.lambda_zip
  etag   = filemd5(var.lambda_zip)
}
