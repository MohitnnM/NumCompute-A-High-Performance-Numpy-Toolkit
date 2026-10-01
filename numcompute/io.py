"""
Responsible for handling raw CSV files, transforming them into appropriate arrays.

This module focuses on providing NumPy-based tools for reading and parsing CSV files using loadtxt or genfromtxt. 
It supports custom delimiters and handling missing values through omission or substitution with placeholder values such as NaN.

The functions in this module form the data ingestion layer of NumCompute, where they read and transform raw CSV files into arrays to be 
passed directly to preprocessing and analysis components. 
"""
