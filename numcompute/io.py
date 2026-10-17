"""
    Responsible for handling numeric CSV files, transforming them into
    appropriate arrays.

    This module focuses on providing NumPy-based tools for reading and
    parsing CSV files using genfromtxt. It supports custom delimiters 
    and handling missing values through substitution with placeholder 
    values such as NaN.

    The functions in this module form the data ingestion layer of 
    NumCompute, where they read and transform raw CSV files into arrays 
    to be passed directly to preprocessing and analysis components. 
"""

import numpy as np


def load_csv(csv_path, file_delimiter=","):

    """
        Loads a CSV file and returns it as an NumPy array.

        This function loads a CSV file and returns its contents as a NumPy 
        array. It parses the data using a specified delimiter, and replaces
        missing values with placeholder NaN values.

        Parameters:
            csv_path: 
                The path to a CSV file being loaded.

            file_delimiter: 
                The character used to separate values in the data.
                Defaults to ",".

        Returns:
            A 2D floating-point NumPy array with shape (n_rows, n_columns).
            Missing values are represented using np.nan.

        Raises: 
            Type Error:
                If file_delimiter is not a string.
            
            ValueError:
                If file_delimiter is not one character or CSV file is empty.

            FileNotFoundError:
                If the specified CSV file does not exist.
        
        Time Complexity:
            O(n) where n is the number of values read from the file.

        Space Complexity:
            O(n) for the returned NumPy array. 
    """

    if not isinstance(file_delimiter, str):
        raise TypeError(f"Delimiter: {file_delimiter} is not string.")

    if len(file_delimiter) != 1:
        raise ValueError("Delimiter must contain exactly one character.")

    csv_data = np.genfromtxt(
        csv_path,
        dtype=float,
        delimiter=file_delimiter,
        filling_values=np.nan,
        ndmin=2
    )

    if csv_data.size == 0:
        raise ValueError("The CSV file is empty.")

    return csv_data
