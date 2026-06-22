import json
import re
from datetime import datetime, timedelta
from typing import List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
import pytz

from payfusion.server.payfusion.apps.integrations.aws.credentials.access_keys import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY_ID,
)
from payfusion.server.payfusion.apps.tools.utilities.available_time_zones import Timezones
from payfusion.server.payfusion.apps.tools.utilities.date_utils import (
    describe_cron_expression,
    convert_to_utc,
    convert_time_via_time_zones,
)


class EventBridgeTool:
    def __init__(self):
        self.aws_access_key_id = AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY_ID

        self.scheduler_client = boto3.client(
            "scheduler",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY_ID,
            region_name="us-east-2",
        )

        self.event_bridge_client = boto3.client(
            "events",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="us-east-2",
        )

    def create_event_rule(self, name, schedule_expression, description=""):
        """
        Create an EventBridge rule.

        :param name: Name of the rule
        :param schedule_expression: Schedule expression (e.g., cron or rate expression)
        :param description: Description of the rule
        :return: Rule ARN
        """
        try:
            response = self.event_bridge_client.put_rule(
                Name=name, ScheduleExpression=schedule_expression, State="ENABLED", Description=description
            )
            return response["RuleArn"]
        except ClientError as e:
            print(f"Error creating rule: {e}")
            return None

    def add_target_to_rule(self, rule_name, target_arn, target_id, input_data=None):
        """
        Add a target to an EventBridge rule.

        :param rule_name: Name of the rule
        :param target_arn: ARN of the target
        :param target_id: ID of the target
        :param input_data: Input data to pass to the target (optional)
        :return: None
        """
        target = {
            "Arn": target_arn,
            "Id": target_id,
        }

        if input_data:
            target["Input"] = json.dumps(input_data)

        try:
            self.event_bridge_client.put_targets(Rule=rule_name, Targets=[target])
            print(f"Target {target_id} added to rule {rule_name}.")
        except ClientError as e:
            print(f"Error adding target: {e}")

    def describe_rule(self, rule_name):
        """
        Describe an EventBridge rule.

        :param rule_name: Name of the rule
        :return: Rule description
        """
        try:
            response = self.event_bridge_client.describe_rule(Name=rule_name)
            return response
        except ClientError as e:
            print(f"Error describing rule: {e}")
            return None

    def delete_rule(self, rule_name):
        """
        Delete an EventBridge rule.

        :param rule_name: Name of the rule
        :return: None
        """
        try:
            # Remove all targets before deleting the rule
            self.event_bridge_client.remove_targets(
                Rule=rule_name, Ids=[target["Id"] for target in self.list_targets(rule_name)]
            )
            self.event_bridge_client.delete_rule(Name=rule_name)
            print(f"Rule {rule_name} deleted.")
        except ClientError as e:
            print(f"Error deleting rule: {e}")

    def list_targets(self, rule_name):
        """
        List targets associated with a rule.

        :param rule_name: Name of the rule
        :return: List of targets
        """
        try:
            response = self.event_bridge_client.list_targets_by_rule(Rule=rule_name)
            return response["Targets"]
        except ClientError as e:
            print(f"Error listing targets: {e}")
            return []

    def standardize_schedule_expression(self, expression: str) -> str:
        """
        Standardizes schedule expressions into consistent, human-readable formats.

        Args:
            expression: Raw schedule expression to standardize

        Returns:
            str: Standardized schedule expression
        """
        if not expression:
            return "No Schedule"

        # Handle expressions like "At 12:30 PM, on the last day of the month"
        if expression.startswith("At"):
            return expression

        # Handle rate expressions like rate(7 days)
        rate_match = re.match(r"rate\((\d+)\s+(\w+)\)", expression.lower())
        if rate_match:
            number = int(rate_match.group(1))
            unit = rate_match.group(2).rstrip("s")  # Remove trailing s if present

            # Map to standardized format
            if unit == "minute":
                if number == 1:
                    return "Every Minute"
                return f"Every {number} Minutes"
            elif unit == "hour":
                if number == 1:
                    return "Hourly"
                return f"Every {number} Hours"
            elif unit == "day":
                if number == 1:
                    return "Daily"
                elif number == 7:
                    return "Weekly"
                elif number == 14:
                    return "Bi-Weekly"
                return f"Every {number} Days"

        # Handle cron expressions
        if expression.startswith("cron"):
            # Add cron parsing logic here if needed
            return expression

        return expression

    def _parse_cadence(self, expression):
        expression = expression.strip()

        if expression.startswith("rate("):
            match = re.match(r"rate\((\d+)\s+(minute|minutes|hour|hours|day|days)\)", expression)
            if not match:
                raise ValueError(f"Unrecognized rate expression format: {expression}")

            value = int(match.group(1))
            unit = match.group(2).rstrip("s")  # singularize for mapping

            unit_map = {
                "minute": timedelta(minutes=value),
                "hour": timedelta(hours=value),
                "day": timedelta(days=value),
            }

            return unit_map[unit]

        elif expression.startswith("cron("):
            raise NotImplementedError("Cron expression parsing not yet supported.")

        else:
            raise ValueError(f"Unknown schedule expression format: {expression}")

    def produce_all_event_dataframe(self):
        """
        Generates a DataFrame containing information about all schedules from the EventBridge client.

        This method retrieves a list of all schedules from the EventBridge client and then iterates
        over them to collect detailed information about each one, including its start date, time,
        time zone, and cadence. The method handles cases where certain schedule details might be
        missing by capturing the names of such schedules in a separate list.

        Returns:
            tuple: A tuple containing two elements:
                - DataFrame: A pandas DataFrame where each row represents a schedule, with columns
                  for schedule name, day of the week, day of the month, hour, minute, time zone,
                  feed time, and cadence. The schedule name is set as the index of the DataFrame.
                - list: A list of schedule names for which details could not be fully retrieved due to
                  missing data.
        """
        data = []
        next_token = None

        while True:
            if next_token:
                response = self.scheduler_client.list_schedules(NextToken=next_token)
            else:
                response = self.scheduler_client.list_schedules()

            # Iterate through the schedules and collect the data
            for schedule in response["Schedules"]:
                schedule_name = schedule["Name"]
                schedule_data = self.scheduler_client.get_schedule(Name=schedule_name)

                def compute_next_invocation(start_date, raw_cadence, now=None):
                    now = now or datetime.now(pytz.UTC)

                    # For rate expressions (e.g. "rate(7 days)")
                    if raw_cadence.startswith("rate"):
                        match = re.match(r"rate\((\d+)\s+(minute|hour|day)s?\)", raw_cadence)
                        if not match:
                            return start_date, start_date

                        value = int(match.group(1))
                        unit = match.group(2)

                        # Calculate the interval between runs
                        delta = timedelta(**{f"{unit}s": value})

                        # If start_date is in the future, that's our next run
                        if start_date > now:
                            return start_date - delta, start_date

                        # Calculate how many intervals have passed since start_date
                        elapsed = now - start_date
                        intervals_passed = elapsed // delta

                        # Next run is start_date + (intervals_passed + 1) * delta
                        next_run = start_date + (intervals_passed + 1) * delta
                        previous_run = next_run - delta

                        return previous_run, next_run

                    # For cron expressions, we'll still return start_date for now
                    # AWS EventBridge uses a specific cron format that would need careful parsing
                    if raw_cadence.startswith("cron"):
                        return start_date, start_date

                    return start_date, start_date

                try:
                    feed_id = schedule_data["Name"]
                    start_date = schedule_data["StartDate"]
                    day_of_week = start_date.strftime("%A")
                    day_of_month = start_date.day
                    hour = start_date.hour
                    minute = str(start_date.minute)
                    if len(minute) == 1:
                        minute = minute + "0"
                    time_zone = start_date.tzname()
                    feed_time = f"{hour}:{minute} {time_zone}"
                    raw_cadence = schedule_data["ScheduleExpression"]
                    cadence = describe_cron_expression(raw_cadence) if raw_cadence.startswith("cron") else raw_cadence
                    standardized_cadence = self.standardize_schedule_expression(str(cadence))
                    last_invocation_time, next_invocation_time = compute_next_invocation(start_date, raw_cadence)

                    # Append a dictionary for each schedule
                    data.append(
                        {
                            "Feed ID": feed_id,
                            "Schedule Name": schedule_name,
                            "Day of Week": day_of_week,
                            "Day of Month": day_of_month,
                            "Start Date": start_date,
                            "Last Invocation Date/Time": last_invocation_time,
                            "Next Invocation Date/Time": next_invocation_time,
                            "Hour": hour,
                            "Minute": minute,
                            "Time Zone": time_zone,
                            "Feed Time": feed_time,
                            "Cadence": cadence,
                            "Standardized Cadence": standardized_cadence,
                            "State": schedule_data["State"],
                        }
                    )
                except KeyError:
                    data.append(
                        {
                            "Feed ID": "",
                            "Schedule Name": f"{schedule_name} Failed to Populate",
                            "Day of Week": "",
                            "Day of Month": "",
                            "Start Date": "",
                            "Last Invocation Date/Time": "",
                            "Next Invocation Date/Time": "",
                            "Hour": "",
                            "Minute": "",
                            "Time Zone": "",
                            "Feed Time": "",
                            "Cadence": "",
                            "Standardized Cadence": "",
                            "State": "",
                        }
                    )

            # Check if there are more results
            next_token = response.get("NextToken")
            if not next_token:
                break

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        df["Cadence"] = [cadence.lstrip("rate(").rstrip(")") for cadence in df["Cadence"]]

        return df

    def create_schedule_expression(
        self,
        schedule_type,
        frequency=None,
        interval=None,
        cron_minute: str = None,
        cron_hour: str = None,
        cron_day_of_month: str = None,
        cron_month: str = None,
        cron_day_of_week: str = None,
        cron_year: str = None,
        time_zone: Timezones = None,
    ):
        """
        Create a schedule expression based on the given parameters.

        :param schedule_type: 'rate' or 'cron'
        :param frequency: 'minute', 'hour', 'day', 'week', 'month', or 'year' (required for 'rate')
        :param interval: integer value for the interval (required for 'rate')
        :param cron_minute:
        :param cron_hour:
        :param cron_day_of_month:
        :param cron_month:
        :param cron_day_of_week:
        :param cron_year:
        :return:
        """

        if schedule_type == "rate":
            valid_frequencies = [
                "minutes",
                "minute",
                "hours",
                "hour",
                "days",
                "day",
                "weeks",
                "week",
                "months",
                "month",
                "years",
                "year",
            ]
            if frequency not in valid_frequencies:
                raise ValueError(f"Invalid frequency: {frequency}. Must be one of {valid_frequencies}.")
            if not isinstance(interval, int) or interval <= 0:
                raise ValueError("Interval must be a positive integer.")
            return f"rate({interval} {frequency})"

        elif schedule_type == "cron":
            converted_to_utc = convert_time_via_time_zones(
                f"{cron_hour}:{cron_minute}", source_timezone=time_zone, target_timezone="UTC"
            )
            cleaned_cron_hour = converted_to_utc.split(":")[0]
            cleaned_cron_minute = converted_to_utc.split(":")[1]
            cron_expression = f"{cleaned_cron_minute} {cleaned_cron_hour} {cron_day_of_month} {cron_month} {cron_day_of_week} {cron_year}"
            if not cron_expression:
                raise ValueError("Cron expression must be provided for schedule_type 'cron'.")
            return f"cron({cron_expression})"

        else:
            raise ValueError("Invalid schedule_type. Must be 'rate' or 'cron'.")

    def create_scheduled_task_in_cluster(
        self,
        rule_name: str = None,
        schedule_expression: str = None,
        commands: List = None,
        state: str = "ENABLED",
    ):
        """
        This method schedules a task to run inside the PayFusion ECS Cluster via EventBridge.

        :param rule_name: This should be, unless otherwise noted, equivalent to the feed id that the schedule pertains to
        :param schedule_expression:
        :param commands:
        :param state: ENABLED or DISABLED
        :return:
        """
        # Create the containerOverrides part with dynamic commands
        container_overrides = {"containerOverrides": [{"name": "PFProdV1", "command": commands}]}

        target = {
            "Arn": "arn:aws:ecs:us-east-2:395720007233:cluster/PFProdV1Cluster",
            "RoleArn": "arn:aws:iam::395720007233:role/ecsEventsRole",
            "EcsParameters": {
                "TaskDefinitionArn": "arn:aws:ecs:us-east-2:395720007233:task-definition/PFProdV1",
                "TaskCount": 1,
                "LaunchType": "FARGATE",
                "NetworkConfiguration": {
                    "awsvpcConfiguration": {
                        "Subnets": [
                            "subnet-03449236a5853163e",
                            "subnet-06d445867f968ddbc",
                            "subnet-0bdc87dc8fe376ee6",
                        ],
                        "SecurityGroups": ["sg-04d0932ff913f2da3"],
                        "AssignPublicIp": "DISABLED",
                    }
                },
                "PlatformVersion": "LATEST",
            },
            "Input": json.dumps(container_overrides),
        }

        # Create the rule in EventBridge
        self.event_bridge_client.put_rule(
            Name=rule_name,
            ScheduleExpression=str(schedule_expression),
            State=state,
            Description=f"{rule_name} to run with scheduled expression: {schedule_expression}.",
        )

        # Set the ECS task as the target for the rule
        response = self.event_bridge_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    "Id": f"{rule_name}-target",
                    "Arn": target["Arn"],
                    "RoleArn": target["RoleArn"],
                    "EcsParameters": target["EcsParameters"],
                    "Input": target["Input"],
                }
            ],
        )

        return response

    def create_scheduled_task(
        self,
        rule_name: str = None,
        schedule_expression: str = None,
        start_date: str = None,
        time_zone: Timezones = None,
        commands: List = None,
        flex_time_window_minutes: int = 15,
        state: str = "ENABLED",
        heavy_task_def: bool = False,
    ):
        """
        This method schedules a task to run inside the PayFusion ECS Service. This is our Prod service and if you're
        attempting to run a task that should not be run on our Prod service, do NOT use this method.

        :param rule_name: This should be, unless otherwise noted, equivalent to the feed id that the schedule pertains to
        :param schedule_expression:
        :param start_date: This should ALWAYS be in the following format: %Y-%m-%d %H:%M
        :param time_zone: You should use the Enum to select YOUR time zone unless you know the time zone for which the schedule should be set to.
        :param commands:
        :param flex_time_window_minutes:
        :param state: ENABLED or DISABLED
        :return:
        """
        expected_date_format = "%Y-%m-%d %H:%M"
        # Convert the datetime to UTC while preserving the date
        dt = datetime.strptime(start_date, expected_date_format)
        if time_zone:
            local_tz = pytz.timezone(time_zone.value)
            dt = local_tz.localize(dt)
            utc_dt = dt.astimezone(pytz.UTC)
            utc_date_str = utc_dt.strftime(expected_date_format)
        else:
            utc_date_str = start_date

        transformed_start_date = datetime.strptime(utc_date_str, expected_date_format).isoformat()

        # Create the containerOverrides part with dynamic commands
        container_overrides = {"containerOverrides": [{"name": "PFProdV1", "command": commands}]}

        task_def_arn = "PFProdV1" if not heavy_task_def else "PFProdV1-Heavy"

        target = {
            "Arn": "arn:aws:ecs:us-east-2:395720007233:cluster/PFProdV1Cluster",
            "RoleArn": "arn:aws:iam::395720007233:role/ecsEventsRole",
            "EcsParameters": {
                "TaskDefinitionArn": f"arn:aws:ecs:us-east-2:395720007233:task-definition/{task_def_arn}",
                "TaskCount": 1,
                "LaunchType": "FARGATE",
                "NetworkConfiguration": {
                    "awsvpcConfiguration": {
                        "Subnets": [
                            "subnet-03449236a5853163e",
                            "subnet-06d445867f968ddbc",
                            "subnet-0bdc87dc8fe376ee6",
                        ],
                        "SecurityGroups": ["sg-04d0932ff913f2da3"],
                        "AssignPublicIp": "DISABLED",
                    }
                },
                "PlatformVersion": "LATEST",
            },
            "Input": json.dumps(container_overrides),
        }

        print(f"Transformed Start Date: {transformed_start_date}")
        response = self.scheduler_client.create_schedule(
            Name=rule_name,
            ScheduleExpression=schedule_expression,
            StartDate=transformed_start_date,
            FlexibleTimeWindow={"Mode": "FLEXIBLE", "MaximumWindowInMinutes": flex_time_window_minutes},
            Target=target,
            State=state,
            Description=f"{rule_name} to run with scheduled expression: {schedule_expression}.",
        )

        return response["ScheduleArn"]

    def update_scheduled_task(
        self,
        feed_id: str,
        new_start_date: str,
        new_schedule_expression: str = None,
        new_status: str = None,
        new_flex_time_window_minutes: int = None,
        time_zone: Timezones = None,
        commands: List = None,
    ):
        """
        This method updates a scheduled task in the PayFusion ECS Service.

        :param feed_id: The feed id (as defined by Pay Fusion)
            Each schedule is named after the feed id for simplicity
        :param new_start_date: The new start date in the format: %Y-%m-%d %H:%M.
        :param new_schedule_expression: The new schedule expression (e.g., cron or rate expression).
        :param new_status: The new status of the schedule, either "ENABLED" or "DISABLED".
        :param time_zone: The time zone of the new start date.
        :return: Response from the update_schedule API call.
        :param commands:
        :return:
        """
        # Fetch the current schedule details
        existing_schedule = self.scheduler_client.get_schedule(Name=feed_id)

        expected_date_format = "%Y-%m-%d %H:%M"
        transformed_start_date = datetime.strptime(
            convert_to_utc(new_start_date, date_format=expected_date_format, input_timezone=time_zone),
            expected_date_format,
        ).isoformat()

        # Update the schedule expression if provided
        if new_schedule_expression:
            schedule_expression = new_schedule_expression
        else:
            schedule_expression = existing_schedule.get("ScheduleExpression")

        # Update the status if provided
        if new_status:
            status = new_status.upper()
        else:
            status = existing_schedule.get("State")

        if new_flex_time_window_minutes:
            flex_time_window_minutes = {"Mode": "FLEXIBLE", "MaximumWindowInMinutes": new_flex_time_window_minutes}
        else:
            flex_time_window_minutes = existing_schedule.get("FlexibleTimeWindow")

        target = existing_schedule.get("Target")
        if commands:
            target = existing_schedule.get("Target")
            container_overrides = {"containerOverrides": [{"name": "PFProdV1", "command": commands}]}
            target["Input"] = json.dumps(container_overrides)

        response = self.scheduler_client.update_schedule(
            Name=feed_id,
            ScheduleExpression=schedule_expression,
            StartDate=transformed_start_date,
            State=status,
            Target=target,
            FlexibleTimeWindow=flex_time_window_minutes,
            Description=f"Updated schedule: {schedule_expression} with new start date: {new_start_date}.",
        )

        return response
