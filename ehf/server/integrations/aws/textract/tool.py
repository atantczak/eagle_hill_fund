import boto3

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class TextractInstance:
    def __init__(self):
        self.aws_access_key_id = AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID

        self.textract_client = boto3.client(
            "textract",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="us-east-2",
        )

    def analyze_document(self, bucket, document_key):
        """
        Analyze a document stored in S3 using Amazon Textract.

        :param bucket: The S3 bucket name.
        :param document_key: The S3 key of the document.
        :return: The response from Textract.
        """
        response = self.textract_client.analyze_document(
            Document={"S3Object": {"Bucket": bucket, "Name": document_key}}, FeatureTypes=["TABLES", "FORMS"]
        )
        return response

    def detect_document_text(self, bucket, document_key):
        """
        Detect text in a document stored in S3 using Amazon Textract.

        :param bucket: The S3 bucket name.
        :param document_key: The S3 key of the document.
        :return: The response from Textract.
        """
        response = self.textract_client.detect_document_text(
            Document={"S3Object": {"Bucket": bucket, "Name": document_key}}
        )
        return response

    def start_document_text_detection(self, bucket, document_key):
        """
        Start asynchronous text detection in a document stored in S3 using Amazon Textract.

        :param bucket: The S3 bucket name.
        :param document_key: The S3 key of the document.
        :return: The job ID for the text detection job.
        """
        response = self.textract_client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": document_key}}
        )
        return response["JobId"]

    def get_document_text_detection(self, job_id):
        """
        Get the results of an asynchronous text detection job.

        :param job_id: The job ID of the text detection job.
        :return: The response from Textract.
        """
        response = self.textract_client.get_document_text_detection(JobId=job_id)
        return response

    def start_document_analysis(self, bucket, document_key):
        """
        Start asynchronous analysis of a document stored in S3 using Amazon Textract.

        :param bucket: The S3 bucket name.
        :param document_key: The S3 key of the document.
        :return: The job ID for the document analysis job.
        """
        response = self.textract_client.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": document_key}}, FeatureTypes=["TABLES", "FORMS"]
        )
        return response["JobId"]

    def get_document_analysis(self, job_id):
        """
        Get the results of an asynchronous document analysis job.

        :param job_id: The job ID of the document analysis job.
        :return: The response from Textract.
        """
        response = self.textract_client.get_document_analysis(JobId=job_id)
        return response
