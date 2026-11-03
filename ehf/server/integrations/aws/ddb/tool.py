import boto3
import logging
import time

import pandas as pd
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO)

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)
from payfusion.server.payfusion.apps.integrations.aws.ddb.utils.utils import dynamodb_obj_list_to_df


class DDB:
    def __init__(self, table_name: str, region: str = "us-east-2"):
        """
        This class is used to connect to a DynamoDB table and perform
        read and write operations on the given table
        """
        self.region = region
        self.access_key = AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID

        self.dynamodb_resource = boto3.resource(
            service_name="dynamodb",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        self.dynamodb_client = boto3.client(
            service_name="dynamodb",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.aws_secret_access_key,
        )

        self.table_name = table_name
        self.table = self.dynamodb_resource.Table(self.table_name)

    @staticmethod
    def populate_item_with_args(**kwargs):
        # Use dict comprehension to create the desired dictionary
        name_value_dict = {name: {"S": str(value)} for name, value in kwargs.items()}

        return name_value_dict

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

    def delete_all_items(self, primary_key_name: str = "feed_id"):
        """
        This method deletes all items in the DynamoDB table.
        """
        try:
            # Scan the table to get all items
            scan_response = self.scan_table()["Items"]

            # Loop through the items and delete them one by one
            for item in scan_response:
                primary_key_value = item[primary_key_name]

                self.dynamodb_client.delete_item(TableName=self.table_name, Key={primary_key_name: primary_key_value})

            logging.info("All items have been deleted from the table.")

        except Exception as e:
            logging.error(f"An error occurred while deleting items from the table. Details: {e}")
            raise e

    def run_between_query(
        self,
        partition_key_col_name,
        partition_key_value,
        column_name,
        first_bound,
        second_bound,
        convert_to_df: bool = True,
    ):
        """
        This method is used to perform a query on the table where the
        returned items are between the first_bound and the second_bound.
        Only values with the given partition key are filtered and returned.

        :param partition_key_col_name: The column name of the partition key
        :param partition_key_value: The value to filter the partition key by
        :param column_name: the name of the column being filtered with the BETWEEN keyword
        :param first_bound: the lower bound used in the BETWEEN operation
        :param second_bound: The upper bound used in the BETWEEN operation
        :param convert_to_df: Default is True
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
            if convert_to_df:
                return dynamodb_obj_list_to_df(response["Items"])
            else:
                return response["Items"]

    def _convert_to_plain_dict(self, items):
        """
        Convert items with DynamoDB data type descriptors to plain values.
        If a value is already a plain type, it is left unchanged.
        """
        plain_items = []
        for item in items:
            plain_item = {}
            for key, value in item.items():
                if isinstance(value, dict) and hasattr(value, "values"):
                    # Try to extract the inner value, assuming DynamoDB type descriptors (e.g., {"S": "value"})
                    try:
                        plain_item[key] = list(value.values())[0]
                    except Exception:
                        plain_item[key] = value
                else:
                    plain_item[key] = value
            plain_items.append(plain_item)
        return plain_items

    def grab_entire_table(self, to_df: bool = False, feed_ids_to_filter_to: list = None) -> pd.DataFrame:
        """
        Retrieve items from the DynamoDB table.
        - If feed_ids_to_filter_to is provided, uses the GSI 'feed_id-index' to query those items.
        - Otherwise, scans the entire table.
        Returns a DataFrame or list of plain dict items.

        :param to_df: If True, returns a DataFrame; otherwise, returns a list of cleaned dict items.
        :param feed_ids_to_filter_to: Optional list of feed_ids to filter by.
        :return: DataFrame or list of cleaned items.
        """
        all_items = []
        initial_delay = 1  # seconds
        max_delay = 32  # seconds

        if feed_ids_to_filter_to:
            if feed_ids_to_filter_to:

                # --- Use query via GSI ---
                for feed_id in feed_ids_to_filter_to:
                    key_condition = Key("feed_id").eq(feed_id)

                    query_kwargs = {
                        "IndexName": "feed_id-index",
                        "KeyConditionExpression": key_condition
                    }

                    while True:
                        try:
                            response = self.table.query(**query_kwargs)
                        except self.dynamodb_resource.meta.client.exceptions.ProvisionedThroughputExceededException:
                            delay = initial_delay
                            while True:
                                time.sleep(delay)
                                try:
                                    response = self.table.query(**query_kwargs)
                                    break
                                except self.dynamodb_resource.meta.client.exceptions.ProvisionedThroughputExceededException:
                                    delay = min(delay * 2, max_delay)

                        all_items.extend(response.get("Items", []))

                        if "LastEvaluatedKey" not in response:
                            break

                        query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        else:
            # --- Fall back to full table scan ---
            scan_kwargs = {"TableName": self.table_name}

            while True:
                try:
                    response = self.dynamodb_client.scan(**scan_kwargs)
                except self.dynamodb_client.exceptions.ProvisionedThroughputExceededException:
                    delay = initial_delay
                    while True:
                        time.sleep(delay)
                        try:
                            response = self.dynamodb_client.scan(**scan_kwargs)
                            break
                        except self.dynamodb_client.exceptions.ProvisionedThroughputExceededException:
                            delay = min(delay * 2, max_delay)

                all_items.extend(response.get("Items", []))
                if "LastEvaluatedKey" not in response:
                    break

                scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        cleaned_items = self._convert_to_plain_dict(all_items)
        return pd.DataFrame(cleaned_items) if to_df else cleaned_items

    def grab_table_w_filter(
        self, 
        attribute_name: str, 
        attribute_value, 
        to_df: bool = False,
        page_size: int = 100,
        timeout_seconds: int = 30,
        projection_expression: str = None,
    ):
        """
        Retrieve items from a DynamoDB table that match a specific attribute value.
        Uses Query when possible (if attribute is a key), falls back to Scan with filter.
        Implements pagination for efficient retrieval of large datasets.

        :param attribute_name: The attribute name to filter by
        :param attribute_value: The value to match
        :param to_df: If True, returns a pandas DataFrame; otherwise, returns a list of dictionaries
        :param page_size: Number of items to retrieve per request (default: 100)
        :param timeout_seconds: Maximum time to wait for the operation (default: 30)
        :param projection_expression: Optional projection expression to limit returned attributes
        :return: A pandas DataFrame or list of dictionaries containing the filtered data
        :raises: TimeoutError if the operation takes too long
        """
        import time
        from botocore.exceptions import ClientError

        table = boto3.resource(
            "dynamodb",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.aws_secret_access_key,
        ).Table(self.table_name)

        # Try to get table description to check if attribute is a key
        try:
            table_desc = table.meta.client.describe_table(TableName=self.table_name)
            key_schema = table_desc['Table']['KeySchema']
            key_attributes = [key['AttributeName'] for key in key_schema]
            
            # Check if we can use Query instead of Scan
            can_use_query = attribute_name in key_attributes

            logging.info(f"Attribute Name: {attribute_name} ... Key Attributes: {key_attributes}")
            
            if can_use_query:
                # Use Query operation
                key_condition = Key(attribute_name).begins_with(attribute_value)
                kwargs = {
                    'KeyConditionExpression': key_condition,
                    'Limit': page_size
                }
                if projection_expression:
                    kwargs['ProjectionExpression'] = projection_expression
                    
                items = []
                start_time = time.time()
                last_evaluated_key = None
                
                while True:
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
                        
                    if last_evaluated_key:
                        kwargs['ExclusiveStartKey'] = last_evaluated_key
                        
                    try:
                        response = table.query(**kwargs)
                        items.extend(response.get('Items', []))
                        
                        if 'LastEvaluatedKey' not in response:
                            break
                            
                        last_evaluated_key = response['LastEvaluatedKey']
                        
                    except table.meta.client.exceptions.ProvisionedThroughputExceededException:
                        time.sleep(1)  # Simple backoff
                        continue
                        
            else:
                # Fall back to Scan with filter
                filter_expression = Attr(attribute_name).eq(attribute_value)
                kwargs = {
                    'FilterExpression': filter_expression,
                    'Limit': page_size
                }
                if projection_expression:
                    kwargs['ProjectionExpression'] = projection_expression
                    
                items = []
                start_time = time.time()
                last_evaluated_key = None
                
                while True:
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
                        
                    if last_evaluated_key:
                        kwargs['ExclusiveStartKey'] = last_evaluated_key
                        
                    try:
                        response = table.scan(**kwargs)
                        items.extend(response.get('Items', []))
                        
                        if 'LastEvaluatedKey' not in response:
                            break
                            
                        last_evaluated_key = response['LastEvaluatedKey']
                        
                    except table.meta.client.exceptions.ProvisionedThroughputExceededException:
                        time.sleep(1)  # Simple backoff
                        continue
                        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise Exception(f"DynamoDB error: {error_code} - {error_message}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

        cleaned_items = self._convert_to_plain_dict(items)
        return pd.DataFrame(cleaned_items) if to_df else cleaned_items
