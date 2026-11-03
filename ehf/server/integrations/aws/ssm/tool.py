import boto3

from payfusion.server.payfusion.apps.integrations.aws.ses.tool import (
    SESClient,
)
from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class SSMClient:
    def __init__(self):
        aws_access_key_id = AWS_ACCESS_KEY_ID
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID
        self.ssm_client = boto3.client(
            "ssm",
            region_name="us-east-2",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.ec2_client = boto3.client(
            "ec2",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        self.ec2_instance_id = "i-0fc93817f94656836"

    def send_command(
        self,
        command,
    ):
        """
        This method runs a command within the given instance id.

        :param command:
        :return:
        """
        try:
            describeInstance = self.ec2_client.describe_instances()
            status = "None"
            for r in describeInstance["Reservations"]:
                for inst in r["Instances"]:
                    if inst["InstanceId"] == self.ec2_instance_id:
                        status = inst["State"]["Name"]
            if status == "running":
                self.ssm_client.send_command(
                    InstanceIds=[self.ec2_instance_id],
                    DocumentName="AWS-RunShellScript",
                    Parameters={"commands": [command]},
                )
                print(f"{command} has sent to {self.ec2_instance_id} successfuly")
            else:
                message = f"Instance {self.ec2_instance_id} not running. Could not complete command."
                print(message)
                SESClient().send_email(
                    message=message,
                    subject=f"Lambda Failure",
                )
        except Exception as e:
            print("Unexpected Exception: %s" % str(e))
        return None
