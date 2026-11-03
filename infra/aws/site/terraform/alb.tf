# One application load balancer: TLS termination, www -> apex, and the Cognito
# login enforced here for the protected paths so the container never handles
# a password or a token exchange.

resource "aws_lb" "site" {
  name                       = var.name
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = aws_subnet.public[*].id
  drop_invalid_header_fields = true
  idle_timeout               = 60
}

resource "aws_lb_target_group" "site" {
  name                 = var.name
  port                 = 8080
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 15

  health_check {
    path                = "/healthz"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.site.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      protocol    = "HTTPS"
      port        = "443"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.site.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate_validation.site.certificate_arn

  # Everything requires a Cognito login; the only exception is the health
  # check (rule below). The app therefore always sees who is looking.
  default_action {
    type = "authenticate-cognito"
    authenticate_cognito {
      user_pool_arn              = aws_cognito_user_pool.site.arn
      user_pool_client_id        = aws_cognito_user_pool_client.alb.id
      user_pool_domain           = aws_cognito_user_pool_domain.site.domain
      scope                      = "openid email profile"
      session_timeout            = var.session_timeout_s
      on_unauthenticated_request = "authenticate"
    }
  }

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.site.arn
  }
}

resource "aws_lb_listener_rule" "public" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 1

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.site.arn
  }

  condition {
    path_pattern {
      values = var.public_paths
    }
  }
}

resource "aws_lb_listener_rule" "www_to_apex" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 5

  action {
    type = "redirect"
    redirect {
      host        = var.domain
      path        = "/#{path}"
      query       = "#{query}"
      protocol    = "HTTPS"
      port        = "443"
      status_code = "HTTP_301"
    }
  }

  condition {
    host_header {
      values = ["www.${var.domain}"]
    }
  }
}
