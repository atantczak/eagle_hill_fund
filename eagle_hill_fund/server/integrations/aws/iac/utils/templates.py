def create_s3_template(
    bucket_name, policy_actions_list: list, effect, principal="*", num_days_till_deletion: int = 365
):
    """
    This method is used to create an s3 bucket with a policy based on the template

    :param bucket_name: the name of the bucket being created
    :param policy_actions_list: a list containing policy actions: E.G. s3:GetObject
    :param effect: The effect that the policy action will give. E.G. Allow
    :param principal: Limits who can view and use the s3 bucket, defaults to everyone
    :return: A dictionary that contains the correctly formatted template
    """
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Template used to create an s3 bucket and a policy",
        "Resources": {
            "S3Bucket": {
                "Type": "AWS::S3::Bucket",
                "Description": "new bucket with policy",
                "Properties": {
                    "BucketName": bucket_name,
                    "LifecycleConfiguration": {"Rules": lifecycle_rule_list(num_days_till_deletion)},
                },
            },
            "S3BucketPolicy": {
                "Type": "AWS::S3::BucketPolicy",
                "Properties": {
                    "Bucket": {"Ref": "S3Bucket"},
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Action": policy_actions_list,
                                "Effect": effect,
                                "Resource": {
                                    "Fn::Join": [
                                        "",
                                        [
                                            "arn:aws:s3:::",
                                            {"Ref": "S3Bucket"},
                                            "/*",
                                        ],
                                    ]
                                },
                                "Principal": principal,
                            }
                        ],
                    },
                },
            },
        },
    }


def create_rest_api_template(description, api_name):
    """
    This method allows for the creation of a rest api template for cloudformation

    :param description: The description that the rest api will have
    :param api_name: The name of the api, see the website for restrictions
    :return: a dictionary that represents the template
    """
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Template used to create a rest api function",
        "Resources": {
            "RestApi": {
                "Type": "AWS::ApiGateway::RestApi",
                "Description": "New Rest API and Policy",
                "Properties": {
                    "Description": description,
                    "Name": api_name,
                },
            }
        },
    }


def create_lambda_template(lambda_name, code, role):
    """
    This method allows for the creation of a lambda function using cloudformation

    :param lambda_name: The name of the lambda function, see website for restrictions
    :param code: This is a dictionary that either includes a link to a zip file or an s3 bucket see:
        https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-lambda-function-code.html
    :param role: The execution role, is an arn.
    :return: The lambda template as a dictionary
    """
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Template used to create a lambda function and its policy",
        "Resources": {
            "LambdaFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": lambda_name,
                    "Code": code,
                    "Role": role,
                },
            }
        },
    }


def create_lambda_with_layer_template(
    layer_name,
    layer_description,
    s3_bucket_name,
    s3_bucket_key,
    lambda_name,
    policies_list,
    runtime: str = "python3.11",
    handler: str = "index.handler",
    runtimes_list: list = ["python3.11"],
):
    """
    This method allows for the creation of a lambda function with a layer included

    :param layer_name: The name of the layer
    :param layer_description: the description of the layer
    :param s3_bucket_name: The s3 bucket that holds the content of the function
    :param s3_bucket_key: the key to the specific file in the s3 bucket for content
    :param lambda_name: The name of the lambda function
    :param policies_list: The list of policies the lambda function can use
    :param runtime: The Python version being used
    :param handler: the handler being used
    :param runtimes_list: The list of runtimes allowed in the layer
    :return: The template as a dictionary
    """
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Template used to create a lambda function with a layer",
        "Resources": {
            "LambdaLayer": {
                "Type": "AWS::Lambda::LayerVersion",
                "Properties": {
                    "LayerName": layer_name,
                    "Description": layer_description,
                    "Content": {
                        "S3Bucket": s3_bucket_name,
                        "S3Key": s3_bucket_key,
                    },
                    "CompatibleRuntimes": runtimes_list,
                },
            },
            "LambdaFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": lambda_name,
                    "Runtime": runtime,
                    "Handler": handler,
                    "Policies": policies_list,
                    "Layers": [{"Ref": "LambdaLayer"}],
                },
            },
        },
    }


def lifecycle_rule_list(num_days):
    """
    This method contains the contents needed for the Rules list for an s3 bucket creation

    :param num_days: the number of days until the deletion of the objects in the s3 bucket
    :return: The list that contains the rules
    """
    return [{"Id": "DeleteOldDataRule", "Status": "Enabled", "ExpirationInDays": num_days, "Prefix": ""}]
