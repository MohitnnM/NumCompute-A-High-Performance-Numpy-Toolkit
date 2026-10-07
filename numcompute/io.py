"""
Responsible for handling raw CSV files, transforming them into appropriate arrays.

This module focuses on providing NumPy-based tools for reading and parsing CSV files using loadtxt or genfromtxt. 
It supports custom delimiters and handling missing values through omission or substitution with placeholder values such as NaN.

The functions in this module form the data ingestion layer of NumCompute, where they read and transform raw CSV files into arrays to be 
passed directly to preprocessing and analysis components. 
"""

import numpy as np


def load_csv(csv_path, file_delimiter=","):

    """
        Loads a CSV file and returns it as an NumPy array.

        This function loads a CSV file and returns its contents as a NumPy array. It parses the data using a specified delimiter, and 
        replaces missing values with placeholder NaN values.

        Parameters:
            csv_path: The path to a CSV file being loaded.
            file_delimiter: The character used to separate values in the data. 

        Returns:
            A NumPy array that contains the parsed data.
    """

    csv_data = np.genfromtxt(
        csv_path, 
        dtype=None,
        names=True,
        delimiter=file_delimiter, 
        missing_values="", 
        filling_values=np.nan #Use NaN to fill in missing numeric values to maintain array shape 
    )

    return csv_data
