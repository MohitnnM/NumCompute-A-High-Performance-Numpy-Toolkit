"""
Compare the performance of Python loop implementations against NumPy
vectorisation.

This module provides benchmarking utilities that measure the execution timing
across multiple runs, compare equivalent implementations, and calculate the
difference in performance, such as speed-up.

To ensure fair comparison, Benchmarking should use the same inputs for both the
Python loop and NumPy vectorisation implementations and verify that both
produce equivalent results before runtime comparison.

Follows the benchmarking philosophies on:
 https://superfastpython.com/python-benchmarking-best-practices/

"""
import gc
from time import perf_counter

import numpy as np

import random

import heapq

import statistics

import numcompute

MODES = ("demo","standard")
DEMO_FUNCS = ("stable_sort", "multi_key_sort", "median", "stan_dev", "imputer")





def benchit(func, *args, repeat=5, mode='standard'):

    """
        Ranks numerical data from 1 to n datapoints.
        Performs a benchmark on a given numcompute function.
        Compares runtime to equivalent Python loops.


        Parameters:
            *args:
                optional arguments for a function if it requires them.

            function:
                The target function to be benchmarked

            repeat:
                How many executions are timed by the benchmark.
                Defaults to 5.

            mode:
                The type of benchmarking to occur:
                - demo: Demonstrates benchmarking utility by comparing the execution time, performance
                    and performance differences between a Numpy implementation of
                    numcompute functions vs a Python loop. 
                    For the most valuable insights, it benchmarks the most performance intensive numcompute
                    functions.
                - standard: standard benchmarking mode that displays execution time

        Returns:
            A printed statement describing execution timing and/or performance
            comparisons for the function and optionally compared to a numpy loop.

        Raises:

            ValueError:
                If a mode for benchmarking that does not exist
                is inputted.

                If in 'demo' mode functions that are being compared do not output the same error.

            TimeoutError:
                Function execution times out at a system level.

            

        Time Complexity:
           Same as algorithm being benchmarked. 

        Space Complexity:
            Same as algorithm being benchmarked
    """
    gc.disable() # Eliminates a potential source of benchmarking variation
                 # by disabling Python garbage collector.
 
    vectorized_benches = np.empty(repeat);  # Initialise array to store repeated benchmark timings.

    if mode not in MODES: # Validating inputted mode
        raise ValueError(
        f"Unknown testing method '{mode}'. Please select from {MODES}."
        )

    match mode:
        case 'demo':   
            
            if func not in DEMO_FUNCS: # Validating inputted function from tuple of demo functions
                raise ValueError(
                f"Unknown demo function '{mode}'. Please select from {DEMO_FUNCS}."
                )
           
            pyloop_benches = np.copy(vectorized_benches) # Creates array to store Python loop implementation benches.

            match func:

                case 'stable_sort':
                    test_arr = [random.randint(1, 50) for _ in range(100)] # Testing input
                    test_arr_sort = sorted(test_arr) # Final output from loop sort (used for verifying)

                    # Verification Step (e.g both vectorized and Python loop functions produce same output)
                    print("\nComparing numcompute stable sort...")
                    np.testing.assert_array_equal(numcompute.stable_sort(test_arr), test_arr_sort)
                    
                    # Filling vectorization bench results
                    for i in range(repeat): 
                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        numcompute.stable_sort(test_arr)  # Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        test_arr_loop = test_arr.copy()
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        test_arr_loop.sort() # Calling the function

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  

                case 'multi_key_sort':

                    #Test inputs
                    test_arr = np.array([
                        [1, 2, 3],
                        [1, 2, 4],
                        [1, 3, 5],
                        [1, 1, 2],
                        [2, 2, 1],
                        [2, 1, 1], 
                        [2, 2, 5]
                    ])

                    test_grid = [
                        [1, 2, 3],
                        [1, 2, 4],
                        [1, 3, 5],
                        [1, 1, 2],
                        [2, 2, 1],
                        [2, 1, 1], 
                        [2, 2, 5]
                    ]

                    # Verification Step (e.g both vectorized and Python loop functions produce same output)
                    print("\nComparing multi key sort..")
                    np.testing.assert_array_equal(numcompute.multi_key_sort(test_arr,[0, 1, 2]), sorted(test_grid, key= lambda x: (x[0], x[1], x[2])))

                    # Filling vectorization bench results
                    for i in range(repeat): 
                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        numcompute.multi_key_sort(test_arr,[0, 1, 2]) # Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        sorted(test_grid, key= lambda x: (x[0], x[1], x[2])) # Calling the function

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  

                case 'median':
                    #Test inputs
                    test_arr = [random.randint(1, 50) for _ in range(100)]

                    # Verification Step (e.g both vectorized and Python loop functions produce same output)
                    print("\nComparing median, vectorized vs python loop functions..")
                    np.testing.assert_allclose(numcompute.median(test_arr), statistics.median(test_arr))

                    for i in range(repeat): 
                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        numcompute.median(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        statistics.median(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  

                case 'stan_dev':
                    #Test inputs
                    test_arr = [random.randint(1, 50) for _ in range(100)]

                    # Verification Step (e.g both vectorized and Python loop functions produce same output)
                    print("\nComparing standard deviation, vectorized vs python loop functions..")
                    np.testing.assert_allclose(numcompute.stan_dev(test_arr), statistics.pstdev(test_arr))

                    for i in range(repeat): 
                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        numcompute.stan_dev(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        statistics.pstdev(test_arr) # Calling the function

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  
                    
                case 'imputer':
                    #Test inputs
                    test_arr = np.full((5,5), np.nan)
                    test_pyloop_verify = test_arr

                    for i in range(len(test_pyloop_verify)):
                        for j in range(len(test_pyloop_verify[i])):
                            test_pyloop_verify[i][j] = -1

                    # Verification Step (e.g both vectorized and Python loop functions produce same output)
                    print("\nComparing imputer, vectorized vs python loop functions..")
                    np.testing.assert_array_equal(numcompute.SimpleImputer(fill_value=-1).fit_transform(test_arr), test_pyloop_verify )

                    for r in range(repeat): 
                        # Refresh test inputs
                        test_arr = np.full((5,5), np.nan)
                        test_pyloop = test_arr

                        ## VECTORISED
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        numcompute.SimpleImputer(fill_value=-1).fit_transform(test_arr)# Calling the function

                        end_time = perf_counter()   # end time

                        vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places


                        ## Python Loop
                        start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                        for i in range(len(test_pyloop)):
                            for j in range(len(test_pyloop[i])):
                                test_pyloop[i][j] = -1   # Performing transform with a loop

                        end_time = perf_counter()   # end time

                        pyloop_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places  

            # Performance table
            print(f'Table of performance (in ms):\n')
            print(f'Pyloop implementation:\n{pyloop_benches}\nVectorization implementation:\n{vectorized_benches}')

            # Print results of comparisons
            print(f'\nAverage python loop implementation execution time: {np.average(pyloop_benches):.6f} ms') # Mean execution time for python loops
            print(f'Average vectorization execution time: {np.average(vectorized_benches):.6f} ms') # Mean execution time

            perf_percentage = (np.average(pyloop_benches) - np.average(vectorized_benches)) / np.average(vectorized_benches) * 100 # Calculates average performance difference from loop to vectorization

            if(perf_percentage < 0):
                 print(f'\nVectorization performance impact is {perf_percentage:.4f}% slow down')
            else:
                 print(f'\nVectorization performance impact is {perf_percentage:.4f}% speed up')

        case 'standard':

            for i in range(repeat):  

                start_time = perf_counter() # Start time, uses a high precision, non-adjustable clock for benchmarking. 

                func(*args)                 # Calling the function

                end_time = perf_counter()   # end time

                vectorized_benches[i] = round((end_time - start_time)*1000, 6) # Add result to the array and round to 6 decimal places
            
            print(f'\nMilisecond execution results for {repeat} repeats (in ms):\n', vectorized_benches) 

            print(f'\nAverage execution time: {np.average(vectorized_benches):.6f} ms') # Mean execution time
 