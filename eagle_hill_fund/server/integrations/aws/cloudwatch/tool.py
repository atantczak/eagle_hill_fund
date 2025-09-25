import logging

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)


class CloudWatchLogger:
    def __init__(
        self,
        logger_name,
        log_stream,
        log_group: str = "payfusion_logs",
        format: str = "%(asctime)s : %(levelname)s - %(message)s",
        region: str = "us-east-2",
    ):
        """
        This class is used to update a given logger to send logs to cloudwatch

        :param logger_name: The name of the logger that will send logs to cloudwatch
        :param log_stream: The name of the logging stream to place each log
        :param log_group: The log group to place each logging stream
        :param format: The format of the messages being sent to the logger
        :param region: The region of AWS to store the logs in
        """
        self.logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(format)
        # This class does not work.
        """
        handler = cloudwatch.CloudwatchHandler(
            log_group=log_group,
            log_stream=log_stream,
            access_id=AWS_ACCESS_KEY_ID,
            access_key=AWS_SECRET_ACCESS_KEY_ID,
            region=region,
        )
        handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        """
