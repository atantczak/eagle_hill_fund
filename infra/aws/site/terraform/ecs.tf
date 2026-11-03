# ECR + one Fargate service behind the ALB. Terraform owns the task definition
# shape (roles, env, logging); GitHub Actions registers a new revision with the
# freshly built image on every deploy, so the service ignores task_definition
# drift here. After changing env/roles in this file, run `terraform apply` and
# then re-run the deploy workflow so the service picks the new revision up.

resource "aws_ecr_repository" "site" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "site" {
  repository = aws_ecr_repository.site.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "site" {
  name              = "/ecs/${var.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_ecs_cluster" "site" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

# Pulls the image, ships logs, and injects the FMP key at launch. Nothing else.
resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  statement {
    sid       = "ReadFmpKey"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [data.aws_secretsmanager_secret.fmp.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "${var.name}-execution-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

# What the running container may do: read the reports bucket (and write the ideas
# box in it), and read (plus write back bars it fetched) the var/ prefix of the
# results bucket. Nothing else.
resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ListReports"
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.reports.arn]
  }
  statement {
    sid       = "ReadReports"
    actions   = ["s3:GetObject"]
    resources = ["${data.aws_s3_bucket.reports.arn}/*"]
  }
  statement {
    sid       = "WriteIdeas"
    actions   = ["s3:PutObject"]
    resources = ["${data.aws_s3_bucket.reports.arn}/ideas/*"]
  }
  statement {
    sid       = "ListVar"
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.var.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.var_prefix}/*"]
    }
  }
  statement {
    sid       = "MirrorVar"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${data.aws_s3_bucket.var.arn}/${var.var_prefix}/*"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${var.name}-task"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

resource "aws_ecs_task_definition" "site" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  # Task-scoped scratch disk (part of the free 20 GiB of ephemeral storage): the
  # lab mirrors var/ into it. Gone when the task stops; refilled from S3.
  volume {
    name = "var"
  }

  # Fargate mounts the scratch volume owned by root, whatever the image says, so a
  # one-shot root container hands it to the app user before "web" starts. It runs
  # the same image (the ":latest" the workflow pushes alongside the SHA tag).
  container_definitions = jsonencode([{
    name      = "init"
    image     = "${aws_ecr_repository.site.repository_url}:latest"
    essential = false
    user      = "0"
    command   = ["chown", "10001:10001", "/var/ehf"]
    mountPoints = [{
      sourceVolume  = "var"
      containerPath = "/var/ehf"
      readOnly      = false
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.site.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "init"
      }
    }
    }, {
    name      = "web"
    image     = "${aws_ecr_repository.site.repository_url}:bootstrap" # replaced by the deploy workflow
    essential = true
    dependsOn = [{ containerName = "init", condition = "SUCCESS" }]
    portMappings = [{
      containerPort = 8080
      protocol      = "tcp"
    }]
    environment = [
      { name = "EHF_SITE_URL", value = local.site_url },
      { name = "EHF_REPORTS_BUCKET", value = data.aws_s3_bucket.reports.id },
      { name = "EHF_COGNITO_DOMAIN", value = local.cognito_domain },
      { name = "EHF_COGNITO_CLIENT_ID", value = aws_cognito_user_pool_client.alb.id },
      { name = "AWS_REGION", value = var.region },
      { name = "PYTHONDONTWRITEBYTECODE", value = "1" },
      # The lab (ehf.web.lab) over a read-only S3 mirror of the Studio's var/.
      { name = "EHF_LAB", value = "on" },
      { name = "EHF_VAR_DIR", value = "/var/ehf" },
      { name = "EHF_VAR_S3_BUCKET", value = data.aws_s3_bucket.var.id },
      { name = "EHF_VAR_S3_PREFIX", value = var.var_prefix },
      { name = "EHF_RESULTS_BACKEND", value = "local" },
    ]
    secrets = [
      { name = "FMP_API_KEY", valueFrom = data.aws_secretsmanager_secret.fmp.arn },
    ]
    mountPoints = [{
      sourceVolume  = "var"
      containerPath = "/var/ehf"
      readOnly      = false
    }]
    readonlyRootFilesystem = true
    linuxParameters        = { initProcessEnabled = true }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.site.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "web"
      }
    }
  }])
}

resource "aws_ecs_service" "site" {
  name                               = var.name
  cluster                            = aws_ecs_cluster.site.id
  task_definition                    = aws_ecs_task_definition.site.arn
  desired_count                      = var.desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 60
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.site.arn
    container_name   = "web"
    container_port   = 8080
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    ignore_changes = [task_definition]
  }

  depends_on = [aws_lb_listener.https]
}
