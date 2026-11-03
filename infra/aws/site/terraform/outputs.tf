output "name_servers" {
  description = "Paste these four into GoDaddy as the domain's nameservers."
  value       = aws_route53_zone.main.name_servers
}

output "url" {
  value = local.site_url
}

output "alb_dns_name" {
  value = aws_lb.site.dns_name
}

output "cognito_hosted_ui" {
  value = "https://${local.cognito_domain}"
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.site.id
}

output "ecr_repository_url" {
  value = aws_ecr_repository.site.repository_url
}

output "deploy_role_arn" {
  value = aws_iam_role.deploy.arn
}

output "ecs" {
  value = "${aws_ecs_cluster.site.name} / ${aws_ecs_service.site.name} (family ${aws_ecs_task_definition.site.family})"
}

output "next_steps" {
  value = <<-EOT
    1. GoDaddy -> DNS -> Nameservers -> "I'll use my own": paste the 4 name_servers above.
       Until they propagate, the certificate validation (and so the HTTPS listener) waits.
    2. Push to the deploy branch or run the "site" workflow in GitHub Actions: it builds
       the image, pushes it to ${aws_ecr_repository.site.repository_url} and rolls the service.
    3. Check your inbox for the Cognito invitation (temporary password), then open ${local.site_url}/library.
    4. python infra/aws/site/ops.py publish-report ...   # upload a write-up to the library
       python infra/aws/site/ops.py status               # service + target health
       python infra/aws/site/ops.py logs                 # tail the container log
  EOT
}
