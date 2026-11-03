# Invite-only user pool. Accounts are declared in var.users; Cognito emails each
# one a temporary password. No self sign-up, email is the username.

resource "aws_cognito_user_pool" "site" {
  name                     = var.name
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]
  deletion_protection      = "ACTIVE"
  mfa_configuration        = var.mfa

  software_token_mfa_configuration {
    enabled = true
  }

  admin_create_user_config {
    allow_admin_create_user_only = true
    invite_message_template {
      email_subject = "Your Eagle Hill Fund sign-in"
      email_message = "You have been invited to eaglehillfund.com. Sign in with {username} and the temporary password {####}; you will be asked to choose your own on first login."
      sms_message   = "Eagle Hill Fund: user {username}, temporary password {####}"
    }
  }

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = false
    temporary_password_validity_days = 7
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  lifecycle {
    ignore_changes = [schema]
  }
}

resource "aws_cognito_user_pool_domain" "site" {
  domain       = "${var.name}-${local.account_id}"
  user_pool_id = aws_cognito_user_pool.site.id
}

resource "aws_cognito_user_pool_client" "alb" {
  name                                 = "${var.name}-alb"
  user_pool_id                         = aws_cognito_user_pool.site.id
  generate_secret                      = true
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]
  supported_identity_providers         = ["COGNITO"]
  callback_urls = [
    "${local.site_url}/oauth2/idpresponse",
    "https://www.${var.domain}/oauth2/idpresponse",
  ]
  logout_urls                   = ["${local.site_url}/"]
  prevent_user_existence_errors = "ENABLED"
  explicit_auth_flows           = ["ALLOW_REFRESH_TOKEN_AUTH", "ALLOW_USER_SRP_AUTH"]
  refresh_token_validity        = 30
  access_token_validity         = 60
  id_token_validity             = 60
  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }
}

resource "aws_cognito_user" "users" {
  for_each = { for u in var.users : u.email => u }

  user_pool_id             = aws_cognito_user_pool.site.id
  username                 = each.value.email
  desired_delivery_mediums = ["EMAIL"]
  attributes = {
    email          = each.value.email
    email_verified = "true"
    name           = each.value.name
  }

  lifecycle {
    ignore_changes = [temporary_password, password]
  }
}
