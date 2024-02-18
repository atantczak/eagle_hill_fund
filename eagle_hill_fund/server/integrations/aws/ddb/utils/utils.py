from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
import pandas as pd


def dynamo_obj_to_python_obj(dynamo_obj: dict) -> dict:
    """
    This method changes the format of a given dynamo dictionary to a
    traditional python dictionary

    :param dynamo_obj: A dictionary returned from DDB query or scan
    :return: the given dictionary in python format
    """
    deserializer = TypeDeserializer()

    return {key: deserializer.deserialize(value) for key, value in dynamo_obj.items()}


def python_obj_to_dynamo_obj(python_obj: dict) -> dict:
    """
    This method changes the format of a traditional python dictionary
    to one in the format that DDB uses

    :param python_obj: A traditional python dictionary
    :return: The given dictionary in ddb format
    """
    serializer = TypeSerializer()

    return {key: serializer.serialize(value) for key, value in python_obj.items()}


def dynamodb_obj_list_to_df(dynamo_obj_list: list) -> pd.DataFrame:
    """
    This method turns a list of dictionary values given from a query
    and turns it into a dataframe

    :param dynamo_obj_list: a list of dictionaries given from a ddb query
    :return: A dataframe version of the given dictionary list
    """
    dict_list = [dynamo_obj_to_python_obj(dict_obj) for dict_obj in dynamo_obj_list]
    return pd.DataFrame(dict_list)