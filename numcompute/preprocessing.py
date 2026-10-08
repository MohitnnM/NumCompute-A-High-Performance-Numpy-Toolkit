"""
Transforms pre-loaded arrays into data suitable for analysis.

This module provides transformations that prepare both numerical and 
categorical data for further analysis or modelling. Components included
are feature standardisation, scaling, and categorical encoding. 

Transformers follow a fit-then-transform style. During fit(X), each 
transformer learns and stores the parameters or categories required for
new data. transform(X) applies the learned state to new data without
relearning it. NumPy vectorisation and broadcasting are used to implement
all numerical operations. 

Classes Included:
  - `StandardScaler` (z-score standardisation)  
  - `MinMaxScaler` (scale to a given range)   
  - `OneHotEncoder` (for categorical variables):  
  
Potential Classes:
  - *(Optional)* Simple imputer to replace NaN with a constant

Implementation requirements:
- Core transformations should be implemented using vectorised NumPy operations,
  not Python loops where possible.
"""

class StandardScaler:
  """
  Standardises numerical features, centres them by subtracting their means
  and scales them by dividing the difference by their standard deviation. 

  APIs implemented:
    - `fit(X) -> self`: calculates and stores the mean and standard 
       deviation per feature.

    - `transform(X) -> X_out`: Uses the stored values from fit(X) to 
       standardise data.

    - `fit_transform(X) -> X_out`: fits the transformer to X, then 
       transforms using the learned state.
  """

class MinMaxScaler:
  """
  Scales numerical data by mapping features into a specified range, using
  the minimum and maximum of each feature.

  APIs implemented:
    - `fit(X) -> self`: stores the minimum and maximum of each feature to 
       learn the range from training data.

    - `transform(X) -> X_out`: Uses the stored values from fit(X) to scale
       data into configured target range.

    - `fit_transform(X) -> X_out`: fits the transformer to X, then transforms
       using the learned state.
  """

class OneHotEncoder:
  """
  Transforms categorical columns into indicator columns for each learned 
  category.
  
  APIs implemented:
    - `fit(X) -> self`: learns and stores the categories for each 
       categorical feature.

    - `transform(X) -> X_out`: Uses the learned categories from fit(X)
       to transform input data into indicator columns for each categorical
       feature.
       
    - `fit_transform(X) -> X_out`: fits the transformer to X, then 
       transforms using the learned state.
  """
