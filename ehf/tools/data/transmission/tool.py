import logging
from abc import ABC, abstractmethod
from typing import Optional

logging.basicConfig(level=logging.INFO)


class DataTransmissionTool(ABC):
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the DataTransmissionTool.

        Args:
            connection_string: String containing connection details for the data source.
                             Format depends on specific implementation.
        """
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
