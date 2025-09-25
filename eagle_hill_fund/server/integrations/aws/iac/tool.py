import boto3
import json

from payfusion.server.payfusion.apps.tools.yaml.tool import YAML
from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)
from payfusion.server.payfusion.apps.integrations.aws.iac.utils.templates import (
    create_s3_template,
    create_lambda_template,
    create_lambda_with_layer_template,
    create_rest_api_template,
)


class CreateAWSResource:
    """
    This class allows for the creation of an AWS Resource. E.G. Lambda, S3, etc..

    :param region: The region the resource will be created in, defaults to 'us-east-2'
    """

    def __init__(self, region: str = "us-east-2"):
        boto3_session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY_ID,
        )
        self.cloud_formation_client = boto3_session.resource("cloudformation", region_name=region)
        self.yaml_obj = YAML()

    def create_resource(self, stack_name, resource_type, yaml_file_name, **kwargs):
        """
        This method allows for the creation of an aws resource through cloudformation

        :param stack_name: The name of the stack being created
        :param resource_type: The type of resource being created must be one of the following: ['s3', 'lambda', 'lambda-w-layer', 'api-gateway']
        :param kwargs: The arguments that are needed for the specific method that the resource will call
        """

        template = None

        if resource_type == "s3":
            template = create_s3_template(**kwargs)
        elif resource_type == "lambda":
            template = create_lambda_template(**kwargs)
        elif resource_type == "lambda-w-layer":
            template = create_lambda_with_layer_template(**kwargs)
        elif resource_type == "api-gateway":
            template = create_rest_api_template(**kwargs)
        else:
            raise Exception(f"Error when trying to determine resource type. Resource type: {resource_type} is invalid")

        self.yaml_obj.write_yaml(yaml_contents=template, file_name=yaml_file_name)

        template = json.dumps(template)
        self.stack_template = template

        self.response = self.cloud_formation_client.create_stack(
            StackName=stack_name,
            TemplateBody=str(template),
        )
