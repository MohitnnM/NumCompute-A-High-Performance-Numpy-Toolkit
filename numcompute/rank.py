"""
Provides ranking and percentile operations for numerical data.

This module computes the rankings and the percentiles. The rankings are
used to preserve the relationship to the original observations, handling
ties by assigning tied values their average rank, the same dense rank,
or distinct ordinal ranks based on original order. 

Percentile is used to identify the value that corresponds to a given position
within an ordered dataset. If the given position falls between two data 
points, then the result is determined using linear, lower, higher, or 
midpoint interpolations. 
"""
import numpy as np

def rank(data, method='average'):
  """
  Ranks numerical data from 1 to n datapoints.

         Parameters:
            data: 
                The numerical data to be ranked, handles tied values.

            method: 
                The ranking method to be performed if values are tied.
                Defaults to "average":
                - average: uses the average rank of the group. 
                - ordinal: orders tied values by some category
                or order. All items receive distinct ranks.
                - dense: unique values are given to distinct values.

        Returns:
            A 1D floating-point NumPy array with shape (n_rows)
            Missing values are represented using np.nan.

        Raises: 
            Type Error:
                If data provided does not have numerical values to
                ranking method.
            
            Value Error:
                If data is empty.

            Name Error: 
                If a method which does not exist for ranking is inputted.
                Method argument is not a string.    
        
        Time Complexity:
           

        Space Complexity:
           
  """
 

  match method:
    case 'average':
        data_ranked_ordinal = np.argsort(data.flatten()).argsort() # Ordinal rankings for fractional/average ranking computations. 
                                                                   # Data is first flattened.
        data_sorted = np.sort(data)                             
        unique, data_counts = np.unique(data_sorted, return_counts='True') # How many times a unique element shows up in the sorted array.
        
        print(data_sorted)
        print(data_counts)
        print("Average ranking:")

    case 'ordinal':
        data_sorted = np.argsort(data.flatten())  # Initial processing of data, ranking only works on sorted data.
                                                  # Data is first flattened.
        data_ranked_ordinal = np.argsort(data_sorted)
        print("Ordinal ranking:",  data_ranked_ordinal + 1)  #Returns ranking using base 1 

    case 'dense':
        data_ranked_dense = np.unique_inverse(data.flatten())  # Sorts unique elements and returns sorted positions that can reconstruct original array
                                                      # Corresponds to a dense ranking, as indices follow pattern of ties sharing positions
                                                      # while next highest element is given the the next position after ties.
                                                      # Data is first flattened.
                                                                    
        print ("Dense ranking:", data_ranked_dense.inverse_indices + 1)   # Returns ranking using base 1 


def percentile(data, q, interpolation):
    """
    Calculates percentiles of any given number in a dataset, for a provided q value.

         Parameters:
            data: 
                sorted, numerical data for which to perform a percentile 
                computation.

            q: 
                The provided percentile. In decimals.
            
            interpolation:
                The kind of percentile which is computed utilizing mathematical
                percentile formula. Has the following interpolations:
                - linear: the default, using linear interpolation formula.
                - lower: lowest data point
                - higher: highest data point
                - midpoint: the average between lowest and highest point.


        Returns:
            Percentile value.
            Missing values are represented using np.nan.

        Raises: 
            Type Error:
                If data provided does not have numerical values to
                compute a percentile with.
            
            ValueError:
                If data is empty.
        
        Time Complexity:
           

        Space Complexity:
           
    """
    match interpolation:
      case 'linear':
        data_flattened = data.flatten()
        data_sorted = np.sort(data_flattened)

        virtual_idx = q * (len(data_sorted) - 1)  # i + g in linear interpolation equation
        lower_idx = int(virtual_idx)  # Is the floor of g
        higher_idx = i + 1             #  i+1 term for final equation.
        fractional = virtual_idx - i  # g in the linear interpolation equation.
        
        a = data_sorted[lower_idx]  # x(i) in percentile equation/
        b = data_sorted[higher_idx]  # x(i + 1) as described in percentile equation.

        percentile = a + (b-a) * fractional
        print("the percentile is: ", percentile)

      case 'lower':
        data_flattened = data.flatten()
        data_sorted = np.sort(data_flattened)

        virtual_idx = q * (len(data_sorted) - 1)  # i + g in linear interpolation equation
        lower_idx = int(virtual_idx)    # Index of the lower interpolation.
        print("the percentile is: ", data_sorted[lower_idx])

      case 'higher':
        data_flattened = data.flatten()
        data_sorted = np.sort(data_flattened)

        virtual_idx = q * (len(data_sorted) - 1)  # i + g in linear interpolation equation
        higher_idx = int(virtual_idx) + 1   # Index of the higher interpolation.
        print("the percentile is: ", data_sorted[higher_idx])

      case 'midpoint':
        data_flattened = data.flatten()
        data_sorted = np.sort(data_flattened)

        virtual_idx = q * (len(data_sorted) - 1)
        lower_idx = int(virtual_idx) # Index of the lower data point.
        higher_idx = lower_idx + 1  # Index of the higher data point.

        percentile_midpoint = (data_sorted[lower_idx] + data_sorted[higher_idx]) / 2    # Midpoint interpolation
        print("the percentile is: ", percentile_midpoint)




# x = np.random.randint(0,10, size=6)    # rank example
x = np.array([0, 2, 3, 2])
print("Initial Array: ", x)    #Unranked array
rank(x, 'average')   
