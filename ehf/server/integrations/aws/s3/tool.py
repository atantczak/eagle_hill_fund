import io
import json
import pickle

import boto3

import pandas as pd
from io import (
    StringIO,
    BytesIO,
)

import pandas.errors

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class s3Instance:
    def __init__(self, non_pf_aws_access_key_id: str = None, non_pf_aws_secret_access_key_id: str = None):
        if non_pf_aws_access_key_id is not None:
            self.aws_access_key_id = non_pf_aws_access_key_id
            self.aws_secret_access_key = non_pf_aws_secret_access_key_id
        else:
            self.aws_access_key_id = AWS_ACCESS_KEY_ID
            self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="us-east-2",
        )
        self.s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

    def upload_pdf(self, pdf_content, bucket, key):
        """
        Uploads a PDF file to S3.

        :param pdf_content: The PDF content to upload.
        :param bucket: The S3 bucket to upload to.
        :param key: The S3 key for the uploaded file.
        :return: None
        """
        try:
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=pdf_content, ContentType="application/pdf")
        except Exception as e:
            raise IOError(f"Could not upload PDF to S3: {e}")

    def check_if_key_exists(self, bucket, key):
        """
        Checks if a specified key exists in a given S3 bucket.

        This function attempts to retrieve metadata for an object using its key within the specified bucket. If the object exists, the function returns True. If the object does not exist or an error occurs during retrieval, it returns False.

        Parameters:
        bucket (str): The name of the S3 bucket where the key will be checked.
        key (str): The key of the object to be checked in the S3 bucket.

        Returns:
        bool: True if the key exists in the bucket, False otherwise.
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False

    def list_contents_in_directory(self, bucket, prefix_to_directory, include_all_object_meta_data: bool = False):
        """

        :param bucket:
        :param prefix_to_directory:
        :return:
        """

        bucket = self.s3_resource.Bucket(bucket)
        objects = bucket.objects.filter(Prefix=prefix_to_directory)

        file_keys = []
        for obj in objects:
            if include_all_object_meta_data:
                file_keys.append(obj)
            else:
                file_keys.append(obj.key)

        return file_keys

    def check_if_file_exists(self, bucket: str = None, prefix_to_directory: str = None, file_name: str = None):
        """
        This method simply checks if the file exists at the specified sftp path.

        :param bucket:
        :param prefix_to_directory:
        :param file_name:
        :return: Boolean
        """
        file_name_with_path = prefix_to_directory + file_name
        directory_contents = self.list_contents_in_directory(bucket=bucket, prefix_to_directory=prefix_to_directory)
        return file_name_with_path in directory_contents

    def get_s3_object(self, bucket, key):
        s3_object = self.s3_resource.Object(
            bucket,
            key,
        )
        file_content = s3_object.get()["Body"].read()
        return file_content

    def create_presigned_url(self, bucket_name, object_name, expiration=28800):
        """
        Generate a presigned URL for an S3 object.

        :param object_name:
        :param expiration: Default - 86400 seconds (1 day)
        :return:
        """

        response = self.s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket_name, "Key": object_name}, ExpiresIn=expiration
        )

        return response

    def upload_json(
        self,
        json_,
        bucket,
        key,
    ):
        """
        Upload a json object to a txt file.

        :param json_:
        :param bucket:
        :param key:
        :return:
        """

        json_as_str = json.dumps(json_)

        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json_as_str,
        )

    def download_json(
        self,
        bucket,
        key,
    ):
        """
        Downloads an s3 bucket's contents in json format

        :param bucket:
        :param key:
        :return:
        """
        file_content = self.get_s3_object(bucket, key).decode("utf-8")
        json_object = json.loads(file_content)

        return json_object

    def upload_dataframe(self, df, bucket, key, end_file_type="csv", header: bool = True):
        """
        Uploads a DataFrame to S3 in CSV, Excel, or Parquet format.

        :param df: DataFrame to upload.
        :param bucket: Bucket to upload to.
        :param key: Key for the uploaded file.
        :param end_file_type: Type of file to create ('csv', 'excel', or 'parquet').
        :param header: Whether to include the header in the file (ignored for parquet).
        :return: None
        """
        # Mapping from file type to buffer type
        file_type_to_buffer_mapping = {
            "csv": StringIO,
            "excel": BytesIO,
            "parquet": BytesIO,
        }

        # Mapping from file type to content type for S3
        content_type_mapping = {
            "csv": "text/csv",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "parquet": "application/octet-stream",  # or 'application/parquet'
        }

        # Create the appropriate buffer
        buffer = file_type_to_buffer_mapping[end_file_type]()

        # Write DataFrame to buffer
        if end_file_type == "csv":
            df.to_csv(buffer, index=False, header=header)
            body = buffer.getvalue()
        elif end_file_type == "excel":
            df.to_excel(buffer, index=False, engine="openpyxl", header=header)
            body = buffer.getvalue()
        elif end_file_type == "parquet":
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            body = buffer.read()
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'excel', or 'parquet'.")

        # Upload to S3
        try:
            self.s3_resource.Object(bucket, key).put(Body=body, ContentType=content_type_mapping[end_file_type])
            buffer.close()
        except Exception as e:
            buffer.close()
            raise IOError(f"Could not upload to S3: {e}")

    def _upload_tempfile(
        self,
        temp_file_name,
        bucket,
        key,
    ):
        """
        Please Note: This method will not actually send data to S3 given the fact that temporary files need to be
        dealt with within the same file that they were created. This method is, therefore, simply held here to illustrate
        how to send a temp file to S3.

        :param temp_file_name:
        :param bucket:
        :param key:
        :return:
        """

        self.s3_client.upload_file(
            temp_file_name,
            bucket,
            key,
        )

    def upload_text_file(self, text_content, bucket, key):
        """
        Uploads a plain text file to S3.

        :param text_content: The text content to upload.
        :param bucket: The S3 bucket to upload to.
        :param key: The S3 key for the uploaded file.
        :return: None
        """
        try:
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=text_content, ContentType="text/plain")
        except Exception as e:
            raise IOError(f"Could not upload text file to S3: {e}")

    def download_file(self, bucket, key, file_type: str = "csv", to_pandas: bool = True, **kwargs):
        """
        Downloads a file from S3 and optionally returns it as a pandas DataFrame or other object.

        :param bucket: S3 bucket name
        :param key: S3 object key
        :param file_type: File format: 'csv', 'xlsx', 'parquet', 'txt', 'fixed_width', 'key', or 'pkl'
        :param to_pandas: Whether to convert to a DataFrame (applies to some formats)
        :param kwargs: Extra arguments for pandas read methods
        :return: DataFrame, object, string, or bytes depending on file_type
        """
        file_content = self.get_s3_object(bucket, key)

        if file_type == "csv":
            try:
                return pd.read_csv(io.StringIO(file_content.decode("utf-8")), **kwargs)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        elif file_type == "xlsx":
            return pd.read_excel(io.BytesIO(file_content), **kwargs)

        elif file_type == "parquet":
            return pd.read_parquet(io.BytesIO(file_content), **kwargs)

        elif file_type == "txt":
            if to_pandas:
                return pd.read_csv(io.StringIO(file_content.decode("utf-8")), **kwargs)
            return file_content.decode("utf-8")

        elif file_type == "fixed_width":
            if to_pandas:
                return pd.read_fwf(io.StringIO(file_content.decode("utf-8")), header=None, **kwargs)
            return file_content

        elif file_type == "key":
            return file_content.decode("utf-8")

        elif file_type == "pkl":
            return pickle.load(io.BytesIO(file_content))

        else:
            raise ValueError("Unsupported file type: choose from [csv, xlsx, parquet, txt, fixed_width, key, pkl]")

    def upload_pickle(self, obj, bucket: str, key: str):
        """
        Pickles a Python object and uploads it to S3.

        :param obj: Any picklable Python object
        :param bucket: S3 bucket name
        :param key: Key (path) where the object will be stored
        """
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)

        self.s3_client.upload_fileobj(buffer, Bucket=bucket, Key=key)
