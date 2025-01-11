import boto3
import logging
from botocore.exceptions import ClientError
from datetime import datetime

logging.basicConfig(level=logging.INFO)

from eagle_hill_fund.server.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)
from eagle_hill_fund.server.integrations.aws.ddb.utils.utils import dynamodb_obj_list_to_df


class DDB:
    def __init__(self, table_name: str, region: str = "us-east-2"):
        """
        This class is used to connect to a DynamoDB table and perform
        read and write operations on the given table
        """
        self.access_key = AWS_ACCESS_KEY_ID
        aws_access_key = AWS_ACCESS_KEY_ID
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID

        self.dynamodb_client = boto3.client(
            service_name="dynamodb",
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
        )

        self.table_name = table_name

    def put_item(self, item: dict):
        """
        This method allows for an item to be placed into a ddb table

        :param item: a dictionary containing the data that will be stored in the table
        :return: true if the operation succeeded, false if it failed
        """
        try:
            self.dynamodb_client.put_item(TableName=self.table_name, Item=item)
            return True
        except Exception as e:
            logging.error(f"an error occured when trying to add the item to the table. Details: {e}")
            return False

    def scan_table(self):
        """
        This method performs a scan on the table, retrieving all items

        :return: The items from the scan
        """
        try:
            table_contents = self.dynamodb_client.scan(TableName=self.table_name)
            return table_contents
        except Exception as e:
            logging.error(f"an error occured when trying to perform a scan on the table. Details: {e}")
            raise (e)

    def run_between_query(self, partition_key_col_name, partition_key_value, column_name, first_bound, second_bound):
        """
        This method is used to perform a query on the table where the
        returned items are between the first_bound and the second_bound.
        Only values with the given partition key are filtered and returned.

        :param partition_key_col_name: The column name of the partition key
        :param partition_key_value: The value to filter the partition key by
        :param column_name: the name of the column being filtered with the BETWEEN keyword
        :param first_bound: the lower bound used in the BETWEEN operation
        :param second_bound: The upper bound used in the BETWEEN operation
        :return: The items in json format
        """
        try:
            response = self.dynamodb_client.query(
                TableName=self.table_name,
                ExpressionAttributeValues={
                    ":p1": {
                        "S": partition_key_value,
                    },
                    ":fb": {
                        "S": first_bound,
                    },
                    ":sb": {
                        "S": second_bound,
                    },
                },
                ExpressionAttributeNames={f"#{column_name}": column_name},
                KeyConditionExpression=f"{partition_key_col_name} = :p1 AND #{column_name} BETWEEN :fb AND :sb",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == "ValidationException":
                logging.warning(
                    f"""There's a validation error. Error message: 
                    {error.response['Error']['Code']} {error.response['Error']['Message']}"""
                )
            else:
                logging.error(
                    f"""couldn't perform the query. here is why: 
                    {error.response['Error']['Code']} {error.response['Error']['Message']}"""
                )
        else:
            return response["Items"]

    def grab_client_table(self, client_name: str = "greenleaf"):
        """

        :param client_name:
        :return:
        """
        date_and_time_now = datetime.now().strftime("%m/%d/%Y:%H:%M:%S")
        response = self.run_between_query(
            partition_key_col_name="client_name",
            partition_key_value=client_name,
            column_name="date",
            first_bound="01/01/2023",
            second_bound=date_and_time_now,
        )
        feed_df = dynamodb_obj_list_to_df(response)
        return feed_df
