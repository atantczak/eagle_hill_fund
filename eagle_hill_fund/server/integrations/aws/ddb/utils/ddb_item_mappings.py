from datetime import datetime, timedelta


def logging_table_item_mapping(
    client_name: str,
    customer_name: str = None,
    feed_name: str = None,
    file_status: bool = None,
    error_details: str = None,
    ttl: int = 365,
):
    """
    This method provides a mapping to be used for creating items in the logging table

    :param client_name: the client name: E.G. greenleaf
    :param customer_name: the name of the clients client: E.G. medmen, trulieve
    :param feed_name: the name of the feed: E.G. wex_cobra, anthem_hsa
    :param file_status: True if the file was successful, and False if it was not
    :param error_details: the details of the error, usually from a try except block
    :param ttl: the time to live for this item. In days
    :return: A dictionary containing everything needed to create an item in the logging database table
    """
    date = datetime.now().strftime("%m/%d/%Y:%H:%M:%S")
    ttl_future = int((datetime.now() + timedelta(days=ttl)).timestamp())

    item_dict = {
        "client_name": {"S": client_name},
        "date": {"S": date},
    }

    if customer_name is not None:
        item_dict["customer_name"] = {"S": customer_name}

    if feed_name is not None:
        item_dict["feed_name"] = {"S": feed_name}

    if file_status is not None:
        if file_status:
            item_dict["file_status"] = {"S": "Success"}
        else:
            item_dict["file_status"] = {"S": "Failed"}

    if error_details is not None:
        item_dict["error_details"] = {"S": error_details}

    item_dict["TTL"] = {"N": str(ttl_future)}

    return item_dict
