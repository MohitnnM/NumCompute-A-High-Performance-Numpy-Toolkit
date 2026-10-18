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
            A 2D floating-point NumPy array with shape (n_rows, n_columns).
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
        data_ranked_ordinal = np.argsort(data).argsort() # Ordinal rankings for fractional/average ranking comptuations.
        # Need to cumsum + mean for each position in the sorted array
        print("Average ranking:", data.argsort().mean())
    case 'ordinal':
        data_sorted = np.argsort(data)  # Initial processing of data, ranking only works on sorted data.
        data_ranked_ordinal = np.argsort(data_sorted)
        print("Ordinal ranking:",  data_ranked_average + 1)  #Returns ranking using base 1 

    case 'dense':
        data_ranked_dense = np.unique_inverse(data)  # Sorts unique elements and returns sorted positions that can reconstruct original array
                                                      # Corresponds to a dense ranking, as indices follow pattern of ties sharing positions
                                                      # while next highest element is given the the next position after ties.
                                                                    
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
                The kind of percentile which is computed.
               - linear: provides the percentile using linear interpolation.
               - lower:  a number for which a provided percentage of values 
                        are lower than.
               - higher: a number for which a provided percentage of values 
                        are higher than.
               - midpoint: provides the percentile corresponding to the
                 50th percentile, i.e 50% of values are above and below.   


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
        print(data_sorted)

        virtual_idx = q * (len(data_sorted) - 1)  # i + g in linear interpolation equation
        i = int(virtual_idx)  # Is the floor of g
        j = i + 1             #  x(i+1) for final equation.
        fractional = virtual_idx - i  # g in the linear interpolation equation.
        
        a = data_sorted[i] 
        b = data_sorted[j]

        percentile = a + (b-a) * fractional
        print("the percentile is: ", percentile)
      case 'lower':
        a
      case 'higher':
        a
      case 'midpoint':
        a

# x = np.random.randint(0,10, size=6)    # rank example
x = np.array([0, 2, 3, 2])
print("Initial Array: ", x)    #Unranked array
rank(x, 'average')   

y = np.array([2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 8, 8, 12, 20])
z = np.array([[10, 7, 4], [3, 2, 1]])

percentile(z, 0.50, 'linear')