from abc import abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict
from tabulate import tabulate


class KVWriter:
    """
    Key Value writer interface. To define a new logger, create a subclass of this one.
    """

    @abstractmethod
    def write(self, key_values: Dict[str, Any], step: int) -> None:
        """
        Write a dictionary to a file
        """

    @abstractmethod
    def close(self) -> None:
        """
        Close owned resources
        """


class PrintOutputFormat(KVWriter):
    def write(self, key_values: Dict[str, Any], step: int) -> None:
        """
        Print all the metrics obtained in the current step
        :param key_values: logged metrics
        :param step: current step
        """
        print(''.join(["-"] * 50))
        print(''.join([" "] * 20 + [f" Step {step} "] + [" "] * 20))
        print(''.join(["-"] * 50))
        for key, value in key_values.items():
            print(str(key).ljust(20), value)
        print(''.join(["\n" * 3]))

    def close(self) -> None:
        pass


class Logger:
    """
    Logger class to allow users set any desired logger.
    Inspired from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/logger.py
    """

    def __init__(self, folder: Optional[str | Path], output_formats: List[KVWriter]):
        """
        :param folder: directory to save the logs if desired
        :param output_formats: list of key-value writers to use
        """
        self.records = defaultdict()
        self.dir = folder
        self.output_formats = output_formats

    @abstractmethod
    def write_param(self, key: str, value: Any):
        """
        Write configuration parameters of the algorithm
        :param key: parameter
        :param value: value
        :return:
        """

    @abstractmethod
    def write_table(self, key: str, table: dict):
        """
        Write metrics in a table
        :param key: Name of the table
        :param table: table to write
        """

    def record(self, key: str, value: Any) -> None:
        """
        Add a specific metric in the records dictionary for this iteration/step.
        :param key: metric to add
        :param value: value of the metric
        """
        self.records[key] = value

    def dump(self, step: int) -> None:
        """
        Write all the metrics for the current iteration/step
        :param step: current step
        """
        # Write logs for current step
        for _format in self.output_formats:
            _format.write(key_values=self.records, step=step)

        # Empty records
        self.records.clear()
