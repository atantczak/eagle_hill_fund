import boto3

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)
from payfusion.server.payfusion.apps.tools.utilities.python_utils import select_random_value


class ECSClient:
    def __init__(self, cluster_name: str = "PFProdV1Cluster"):
        self.aws_access_key_id = AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID

        self.ecs_client = boto3.client(
            "ecs",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="us-east-2",
        )

        self.cluster_name = cluster_name

    def execute_command_in_task(self, commands: list = None, task_definition: str = "PFProdV1"):
        response = self.ecs_client.run_task(
            cluster="PFProdV1Cluster",
            launchType="FARGATE",
            taskDefinition=task_definition,
            platformVersion="LATEST",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": ["subnet-06d445867f968ddbc", "subnet-0bdc87dc8fe376ee6", "subnet-03449236a5853163e"],
                    "securityGroups": ["sg-04d0932ff913f2da3"],
                    "assignPublicIp": "DISABLED",
                }
            },
            overrides={"containerOverrides": [{"name": "PFProdV1", "command": commands}]},
        )
        return response

    def list_running_tasks(self):
        try:
            response = self.ecs_client.list_tasks(cluster=self.cluster_name, desiredStatus="RUNNING")
            return response["taskArns"]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def describe_tasks(self, task_arns):
        try:
            response = self.ecs_client.describe_tasks(
                cluster=self.cluster_name, tasks=task_arns if isinstance(task_arns, list) else [task_arns]
            )
            return response["tasks"]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def enable_execute_command_in_service(self):
        try:
            response = self.ecs_client.update_service(
                cluster=self.cluster_name,
                service="PFOneOffTaskService",
                taskDefinition="PFProdV1:44",
                enableExecuteCommand=True,
                forceNewDeployment=True,
            )
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def list_clusters(self):
        response = self.ecs_client.list_clusters()
        clusters = response.get("clusterArns", [])
        return clusters

    def check_task_usage(self, task_arn):
        try:
            task_details = self.describe_tasks(self.cluster_name, [task_arn])
            if not task_details:
                return None

            # Assuming that you have a way to determine if the task is in use
            # For example, by checking specific logs, metrics, or other attributes
            task = task_details[0]
            container_statuses = task["containers"]

            for container in container_statuses:
                if container["lastStatus"] == "RUNNING":
                    # Add additional checks as necessary
                    print(f"Container {container['name']} is running.")
                    # Check CPU, memory usage, etc., if needed

            return task_details
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
