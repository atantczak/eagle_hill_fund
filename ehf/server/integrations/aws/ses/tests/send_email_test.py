from payfusion.server.payfusion.apps.integrations.aws.ses.tool import SESClient
from payfusion.server.payfusion.apps.tools.utilities.date_utils import todays_date

SESClient().send_email(
    recipient="andrew@benefitcloud.io",
    subject="Test Email from PF Codebase",
    message=f"Hello! The date is: {todays_date()}",
)
