# Route 53 is authoritative for the domain once its name servers are pasted
# into the registrar (GoDaddy). The certificate is validated by DNS inside the
# same zone, so renewals are automatic.

resource "aws_route53_zone" "main" {
  name    = var.domain
  comment = "Eagle Hill Fund site (terraform: infra/aws/site)"
}

resource "aws_acm_certificate" "site" {
  domain_name               = var.domain
  subject_alternative_names = ["*.${var.domain}"]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.site.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  zone_id         = aws_route53_zone.main.zone_id
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
  allow_overwrite = true # apex and wildcard share one validation record
}

# Blocks until the registrar points at Route 53 and ACM sees the records.
resource "aws_acm_certificate_validation" "site" {
  certificate_arn         = aws_acm_certificate.site.arn
  validation_record_fqdns = [for r in aws_route53_record.cert_validation : r.fqdn]

  timeouts {
    create = "4h"
  }
}

resource "aws_route53_record" "apex" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain
  type    = "A"
  alias {
    name                   = aws_lb.site.dns_name
    zone_id                = aws_lb.site.zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.${var.domain}"
  type    = "A"
  alias {
    name                   = aws_lb.site.dns_name
    zone_id                = aws_lb.site.zone_id
    evaluate_target_health = false
  }
}
