"""
Contains shared numerical utility functions used by other modules 
throughout the NumCompute package.

This is mainly created to house reusable helper functions that support
the modules in the NumCompute package, including functions such as 
distance calculation and activation functions. It may also include 
numerically stable operations like log-sum-exp, batching and shared 
top-k utilities. 

The purpose of this module is to provide the NumCompute packages with
common numerical operations that do not belong exclusively to one 
module. Where applicable, the implementation will use NumPy
vectorisation and numerical stability. 
"""
