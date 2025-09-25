import boto3
import json
from botocore.exceptions import (
    ClientError,
)

from payfusion.server.payfusion.apps.data_pipelines.enums.secret_enums import PFSecretSoftwareTypeEnum, PFSecretTypeEnum
from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class SMClient:
    def __init__(self):
        aws_access_key_id = AWS_ACCESS_KEY_ID
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID
        self.sm_client = boto3.client(
            service_name="secretsmanager",
            region_name="us-east-2",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        self.secret_json = None

    def list_all_secrets(self):
        """
        This method lists all secrets in the current AWS region.

        :return: A list of all secrets
        """
        secrets = []
        try:
            paginator = self.sm_client.get_paginator("list_secrets")
            for page in paginator.paginate():
                secrets.extend(page["SecretList"])
        except ClientError as e:
            raise e

        secret_names = []
        for secret in secrets:
            secret_names.append(secret["Name"])
        return secret_names

    def get_secret(self, secret_name, multiple_secrets: bool = True):
        """
        This method connects to aws secrets manager and retrieves a
        secret that has the same name as the given parameter

        :param secret_name: The name of the secret being retrieved
        :param multiple_secrets: DEFAULT IS TRUE. Please Note: This was changed from False to True
            on February 2nd, 2024.
        :return: the secret's value
        :return:
        """
        try:
            get_secret_value_response = self.sm_client.get_secret_value(SecretId=secret_name)
        except ClientError as e:
            raise e

        secret = get_secret_value_response["SecretString"]

        secret_json = json.loads(secret)

        self.secret_json = secret_json  # Stored for manual retrieval if needed.

        if multiple_secrets:
            return secret_json
        else:
            return secret_json["secret"]

    def _enforce_key_format_and_enums(self, key):
        """
        This method enforces the key format and enums for the keys in the secret.
        If it is a "master" secret, it will have 6 parts, otherwise it will have 5 parts.

        Regular Secret Format:
            {child_client}_{incoming/outgoing}_{vendor}_{software_type}_{secret_type}

        Master Secret Format:
            {parent_client}_{incoming/outgoing}_{vendor}_{software_type}_master_{secret_type}

        :param key: The key to be validated
        :return: The validated key
        """
        key_parts = key.split("_")
        num_parts = 6 if "master" in key else 5
        if len(key_parts) != num_parts:
            raise ValueError(f"The key ({key}) does not have the correct number of parts.")

        incoming_outgoing = key_parts[1]
        software_type = key_parts[3]
        secret_type = key_parts[5] if "master" in key else key_parts[4]

        if incoming_outgoing not in ["incoming", "outgoing"]:
            raise ValueError(f"The incoming/outgoing part of the key '{incoming_outgoing}' is not valid.")
        if PFSecretSoftwareTypeEnum(software_type) not in PFSecretSoftwareTypeEnum:
            raise ValueError(f"The software type '{software_type}' is not a valid software type.")
        if PFSecretTypeEnum(secret_type) not in PFSecretTypeEnum:
            raise ValueError(f"The secret type '{secret_type}' is not a valid secret type.")

    def create_secret(self, name: str, secret_dict) -> None:
        """
        This method connects to aws secrets manager and creates a new secret
        with the given name, keys, and values.

        :param name: The name of the secret being created
        :param secret_dict:
        :return:
        """
        # Apply self._enforce_key_format_and_enums to all keys in secret_dict
        for key in secret_dict.keys():
            self._enforce_key_format_and_enums(key)

        try:
            self.sm_client.create_secret(Name=name, SecretString=json.dumps(secret_dict))
        except ClientError as e:
            raise e

    def update_secret(self, name: str, keys: list[str], values: list[str], force_formatting_check: bool = True) -> None:
        """
        This method updates an existing secret in AWS Secrets Manager. It fetches
        the current secret, updates it with the new keys and values, and then saves
        the updated secret.

        :param name:
        :param keys:
        :param values:
        :param force_formatting_check:
        :return:
        """
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same number of elements")

        if force_formatting_check:
            for key in keys:
                self._enforce_key_format_and_enums(key)

        try:
            # Fetch the current secret
            current_secret = self.sm_client.get_secret_value(SecretId=name)
            secret_dict = json.loads(current_secret["SecretString"])

            # Update the secret with new keys and values
            for i in range(len(keys)):
                secret_dict[keys[i]] = values[i]

            # Update the secret in Secrets Manager
            self.sm_client.update_secret(SecretId=name, SecretString=json.dumps(secret_dict))
        except self.sm_client.exceptions.ResourceNotFoundException:
            raise ValueError(f"The secret with name '{name}' does not exist.")
        except ClientError as e:
            raise e
