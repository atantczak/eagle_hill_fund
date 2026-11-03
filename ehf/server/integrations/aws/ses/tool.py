import boto3
import os
import pandas as pd
from email.mime.multipart import (
    MIMEMultipart,
)
from email.mime.text import (
    MIMEText,
)
from email.mime.application import (
    MIMEApplication,
)

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class SESClient:
    def __init__(self):
        aws_access_key_id = AWS_ACCESS_KEY_ID
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID
        self.ses_client = boto3.client(
            "ses",
            region_name="us-east-2",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def send_email(
        self,
        sender: str = "info@benefitcloud.io",
        recipient: str = "info@benefitcloud.io",
        message: str = "",
        html_message: str = None,
        subject: str = "Lambda Update",
    ):
        """
        Send an email using AWS SES.

        :param sender: Email address of the sender.
        :param recipient: Email address of the recipient.
        :param text_message: The plain text message content.
        :param html_message: The HTML message content.
        :param subject: The subject of the email.
        :return: None
        """
        # Construct the body of the message
        body = {}
        if message:
            body["Text"] = {"Data": message}
        if html_message:
            body["Html"] = {"Data": html_message}

        # Send the email
        try:
            self.ses_client.send_email(
                Source=sender,
                Destination={
                    "ToAddresses": [recipient],
                },
                Message={
                    "Subject": {
                        "Data": subject,
                    },
                    "Body": body,
                },
            )
        except Exception as e:
            print(f"Error: {e}")

    def send_email_with_attachment_csv(
        self,
        sender: str = "info@benefitcloud.io",
        recipient: str = "info@benefitcloud.io",
        message: str = "",
        subject: str = "Lambda Update",
        df: pd.DataFrame = None,
        filename: str = "data.csv",
    ):
        """
        :param sender:
        :param recipient:
        :param message:
        :param df:
        :param filename:
        :return:
        """
        # Create a MIME multipart message
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject

        # Add the message body
        msg.attach(MIMEText(message))

        # If a DataFrame is provided, attach it as a CSV file
        if df is not None:
            # Convert the DataFrame to a CSV string
            csv_string = df.to_csv(index=False)

            # Create a MIME application message for the CSV file
            csv_part = MIMEApplication(
                csv_string.encode(),
                Name=filename,
            )
            csv_part["Content-Disposition"] = f'attachment; filename="{filename}"'

            # Add the CSV file to the message
            msg.attach(csv_part)

        # Send the email using the SES client
        self.ses_client.send_raw_email(RawMessage={"Data": msg.as_string()})

    def create_multipart_message(
        self, sender: str, recipients: list, subject: str, text: str = None, html: str = None, attachments: list = None
    ) -> MIMEMultipart:
        """
        Creates a MIME multipart message object.
        Uses only the Python email standard library.

        :param sender: the Sender of the email
        :param recipients: list of recipients
        :param subject: the subject of the email
        :param text: the text version of the email body
        :param html: The html version of the email body
        :param attachments: list of files attached to the email
        :return: A 'MIMEMultipart' to be used to send the email
        """
        multipart_content_subtype = "alternative" if text and html else "mixed"
        msg = MIMEMultipart(multipart_content_subtype)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)

        if text:
            part = MIMEText(text, "plain")
            msg.attach(part)
        if html:
            part = MIMEText(html, "html")
            msg.attach(part)

        for attachment in attachments or []:
            with open(attachment, "rb") as f:
                part = MIMEApplication(f.read())
                part.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachment))
                print(os.path.basename(attachment))
                msg.attach(part)

        return msg

    def send_email_with_attachment(
        self,
        subject: str,
        sender: str = "info@benefitcloud.io",
        recipients: list = ["info@benefitcloud.io"],
        text: str = None,
        html: str = None,
        attachments: list = None,
    ) -> dict:
        """
        This method sends an email with an attachment to a subscribed list of emails

        :param sender: the Sender of the email
        :param recipients: list of recipients
        :param subject: the subject of the email
        :param text: the text version of the email body
        :param html: The html version of the email body
        :param attachments: list of files attached to the email
        """
        try:
            msg = self.create_multipart_message(sender, recipients, subject, text, html, attachments)
        except Exception as e:
            raise Exception(f"an error occured when try to create the message. details: {e}")

        try:
            response = self.ses_client.send_raw_email(RawMessage={"Data": msg.as_string()})
        except Exception as e:
            raise Exception(f"an error occured when trying to send the email. Details {e}")

        return response
