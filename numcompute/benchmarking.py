"""
Compare the performance of Python loop implementations against NumPy
vectorisation.

This module provides benchmarking utilities for NumCompute batch and streaming
implementation.

It measures the execution timing across multiple runs, compare equivalent
implementations, and calculate the difference in performance, such as
speed-up. It also benchmarks streaming components, including incremental
StandardScaler updates and DecisionTreeClassfier performance across
different buffer sizes.

To ensure fair comparison, Benchmarking should use the same inputs for both the
Python loop and NumPy vectorisation implementations and verify that both
produce equivalent results before runtime comparison.

Follows the benchmarking philosophies on:
 https://superfastpython.com/python-benchmarking-best-practices/

"""

import gc
import random
import statistics
from time import perf_counter

import numpy as np

import numcompute

MODES = ("demo", "standard")
DEMO_FUNCS = ("stable_sort", "multi_key_sort", "median", "stan_dev", "imputer")


def benchit(func, *args, repeat=5, mode="standard"):
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
            - demo: Demonstrates benchmarking utility by comparing the execution
              time, performance and performance differences between a Numpy
              implementation of numcompute functions vs a Python loop. For the
              most valuable insights, it benchmarks the most performance intensive
              numcompute functions.
            - standard: standard benchmarking mode that displays execution time.

    Returns:
        A printed statement describing execution timing and/or performance
        comparisons for the function and optionally compared to a numpy loop.

    Raises:

        ValueError:
            If a mode for benchmarking that does not exist
            is inputted.

            If in 'demo' mode functions that are being compared do not output
            the same error.

        TimeoutError:
            Function execution times out at a system level.



    Time Complexity:
       Same as algorithm being benchmarked.

    Space Complexity:
        Same as algorithm being benchmarked
    """
    gc.disable()  # Eliminates a potential source of benchmarking variation
    # by disabling Python garbage collector.

    vectorised_benches = np.zeros(
        repeat
    )  # Initialise array to store repeated benchmark timings.

    if mode not in MODES:  # Validating inputted mode
        raise ValueError(
            f"Unknown testing method '{mode}'. Please select from {MODES}."
        )

    match mode:
        case "demo":
            if (
                func not in DEMO_FUNCS
            ):  # Validating inputted function from tuple of demo functions
                raise ValueError(
                    f"Unknown demo function '{mode}'. Please select from {DEMO_FUNCS}."
                )

            pyloop_benches = np.copy(
                vectorised_benches
            )  # Creates array to store Python loop implementation benches.

            match func:
                case "stable_sort":
                    test_arr_verify = [
                        random.randint(1, 50) for _ in range(100)
                    ]  # Testing input
                    test_arr_sort_verify = sorted(
                        test_arr_verify
                    )  # Final output from loop sort (used for verifying)

                    # Verification Step (e.g both vectorised and Python loop
                    # functions produce same output)
                    print("\nComparing numcompute stable sort...")
                    np.testing.assert_array_equal(
                        numcompute.stable_sort(test_arr_verify), test_arr_sort_verify
                    )

                    # Filling vectorization bench results
                    for i in range(repeat):
                        # Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]
                        test_arr_loop = test_arr.copy()

                        ## VECTORISED
                        # Start time, uses a high precision, non-adjustable clock
                        # for benchmarking.
                        start_time = perf_counter()
                        numcompute.stable_sort(test_arr)  # Calling the function

                        end_time = perf_counter()  # end time

                        vectorised_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        # Start time, uses a high precision, non-adjustable clock
                        # for benchmarking.
                        start_time = perf_counter()

                        test_arr_loop.sort()  # Calling the function

                        end_time = perf_counter()  # end time

                        pyloop_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                case "multi_key_sort":
                    # Test inputs
                    test_arr_verify = np.array(
                        [
                            [1, 2, 3],
                            [1, 2, 4],
                            [1, 3, 5],
                            [1, 1, 2],
                            [2, 2, 1],
                            [2, 1, 1],
                            [2, 2, 5],
                        ]
                    )

                    test_grid_verify = [
                        [1, 2, 3],
                        [1, 2, 4],
                        [1, 3, 5],
                        [1, 1, 2],
                        [2, 2, 1],
                        [2, 1, 1],
                        [2, 2, 5],
                    ]

                    # Verification Step (e.g both vectorised and Python loop
                    # functions produce same output)
                    print("\nComparing multi key sort..")
                    np.testing.assert_array_equal(
                        numcompute.multi_key_sort(test_arr_verify, [0, 1, 2]),
                        sorted(test_grid_verify, key=lambda x: (x[0], x[1], x[2])),
                    )

                    # Filling vectorization bench results
                    for i in range(repeat):
                        # Refresh inputs
                        test_arr = np.array(
                            [
                                [1, 2, 3],
                                [1, 2, 4],
                                [1, 3, 5],
                                [1, 1, 2],
                                [2, 2, 1],
                                [2, 1, 1],
                                [2, 2, 5],
                            ]
                        )

                        test_grid = [
                            [1, 2, 3],
                            [1, 2, 4],
                            [1, 3, 5],
                            [1, 1, 2],
                            [2, 2, 1],
                            [2, 1, 1],
                            [2, 2, 5],
                        ]

                        ## VECTORISED
                        # Start time, uses a high precision, non-adjustable
                        # clock for benchmarking.
                        start_time = perf_counter()
                        numcompute.multi_key_sort(
                            test_arr, [0, 1, 2]
                        )  # Calling the function

                        end_time = perf_counter()  # end time

                        vectorised_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        # Start time, uses a high precision, non-adjustable clock
                        # for benchmarking.
                        start_time = perf_counter()

                        sorted(
                            test_grid, key=lambda x: (x[0], x[1], x[2])
                        )  # Calling the function

                        end_time = perf_counter()  # end time

                        pyloop_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                case "median":
                    # Test inputs
                    test_arr_verify = [random.randint(1, 50) for _ in range(100)]

                    # Verification Step (e.g both vectorised and Python loop
                    # functions produce same output)
                    print("\nComparing median, vectorised vs python loop functions..")
                    np.testing.assert_allclose(
                        numcompute.median(test_arr_verify),
                        statistics.median(test_arr_verify),
                    )

                    for i in range(repeat):
                        # Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]

                        ## VECTORISED
                        # Start time, uses a high precision, non-adjustable clock
                        #  for benchmarking.
                        start_time = perf_counter()

                        numcompute.median(test_arr)  # Calling the function

                        end_time = perf_counter()  # end time

                        vectorised_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        # Start time, uses a high precision, non-adjustable clock
                        # for benchmarking.
                        start_time = perf_counter()

                        statistics.median(test_arr)  # Calling the function

                        end_time = perf_counter()  # end time

                        pyloop_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                case "stan_dev":
                    # Test inputs
                    test_arr_verify = [random.randint(1, 50) for _ in range(100)]

                    # Verification Step (e.g both vectorised and Python loop
                    # functions produce same output)
                    print(
                        "\nComparing standard deviation, vectorised vs python"
                        " loop functions."
                    )
                    np.testing.assert_allclose(
                        numcompute.stan_dev(test_arr_verify),
                        statistics.pstdev(test_arr_verify),
                    )

                    for i in range(repeat):
                        # Refresh inputs
                        test_arr = [random.randint(1, 50) for _ in range(100)]

                        ## VECTORISED
                        # Start time, uses a high precision, non-adjustable clock
                        #  for benchmarking.
                        start_time = perf_counter()
                        numcompute.stan_dev(test_arr)  # Calling the function

                        end_time = perf_counter()  # end time

                        vectorised_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        # Start time, uses a high precision, non-adjustable clock
                        #  for benchmarking.
                        start_time = perf_counter()

                        statistics.pstdev(test_arr)  # Calling the function

                        end_time = perf_counter()  # end time

                        pyloop_benches[i] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                case "imputer":
                    # Test inputs
                    test_arr_verify = np.full((5, 5), np.nan)
                    test_pyloop_verify = test_arr_verify.copy()

                    for i in range(len(test_pyloop_verify)):
                        for j in range(len(test_pyloop_verify[i])):
                            test_pyloop_verify[i][j] = -1

                    # Verification Step (e.g both vectorised and Python loop
                    # functions produce same output)
                    print("\nComparing imputer, vectorised vs python loop functions.")
                    np.testing.assert_array_equal(
                        numcompute.SimpleImputer(fill_value=-1).fit_transform(
                            test_arr_verify
                        ),
                        test_pyloop_verify,
                    )

                    for r in range(repeat):
                        # Refresh inputs
                        test_arr = np.full((5, 5), np.nan)
                        test_pyloop = test_arr.copy()

                        ## VECTORISED
                        # Start time, uses a high precision, non-adjustable clock
                        #  for benchmarking.
                        start_time = perf_counter()

                        numcompute.SimpleImputer(fill_value=-1).fit_transform(
                            test_arr
                        )  # Calling the function

                        end_time = perf_counter()  # end time

                        vectorised_benches[r] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

                        ## Python Loop
                        # Start time, uses a high precision, non-adjustable clock
                        #  for benchmarking.
                        start_time = perf_counter()

                        for i in range(len(test_pyloop)):
                            for j in range(len(test_pyloop[i])):
                                test_pyloop[i][
                                    j
                                ] = -1  # Performing transform with a loop
                        end_time = perf_counter()  # end time

                        pyloop_benches[r] = round(
                            (end_time - start_time) * 1000, 6
                        )  # Add result to the array and round to 6 decimal places

            # Performance table
            print("Table of performance (in ms):\n")
            print(
                f"Pyloop implementation:\n{pyloop_benches}"
                f"\nVectorization implementation:\n{vectorised_benches}"
            )

            # Print results of comparisons
            print(
                "\nAverage python loop implementation execution time:"
                f" {np.average(pyloop_benches):.6f} ms"
            )  # Mean execution time for python loops
            print(
                "Average vectorization execution time:"
                f" {np.average(vectorised_benches):.6f} ms"
            )  # Mean execution time

            pyloop_avg_time = np.average(pyloop_benches)
            vectorised_avg_time = np.average(vectorised_benches)

            # Calculates average performance difference from loop to vectorization
            perf_percentage = (
                (vectorised_avg_time - pyloop_avg_time) / pyloop_avg_time * 100
            )

            if perf_percentage > 0:
                print(
                    "\nVectorization performance impact is"
                    f" {perf_percentage:.4f}% slow down"
                )
            else:
                print(
                    "\nVectorization performance impact is"
                    f" {abs(perf_percentage):.4f}% speed up"
                )

        case "standard":
            for i in range(repeat):
                # Start time, uses a high precision, non-adjustable clock
                #  for benchmarking.
                start_time = perf_counter()

                func(*args)  # Calling the function

                end_time = perf_counter()  # end time

                vectorised_benches[i] = round(
                    (end_time - start_time) * 1000, 6
                )  # Add result to the array and round to 6 decimal places

            print(
                f"\nMilisecond execution results for {repeat} repeats (in ms):\n",
                vectorised_benches,
            )

            print(
                f"\nAverage execution time: {np.average(vectorised_benches):.6f} ms"
            )  # Mean execution time


def _loop_streaming_scaler_stats(X, size_of_chunk):
    """
    This helper function will calculate the running feature means and
    the standard deviation using python loops.

    It provides a loop-based equivalent of the running statistics that
    StandardScaler.partial_fit() maintain, which allows for its streaming
    implementation to be benchmarked against a non-vectoriese approach.

    Parameters:
        X:
            A two-dimensional numerical input data.

        size_of_chunk:
            The number of observations that are processed in each of
            the streaming chunks.

    Returns:
        A tuple that contains two arrays with the cumulative feature
        means and population standard deviations.

    Time Complexity:
        O(n * f); n is the number of observations; f is the number
        of features.

    Space Complexity:
        O(f)
    """

    n_features = X.shape[1]
    n_samples = X.shape[0]
    counts = [0] * n_features
    means = [0.0] * n_features
    M2 = [0.0] * n_features

    for origin in range(0, n_samples, size_of_chunk):
        chunk = X[origin : origin + size_of_chunk]

        for row in chunk:
            for feature in range(n_features):
                value = row[feature]

                counts[feature] += 1

                delta = value - means[feature]
                means[feature] += delta / counts[feature]

                delta2 = value - means[feature]
                M2[feature] += delta * delta2

    scales = [0.0] * n_features

    for feature in range(n_features):
        var = M2[feature] / counts[feature]
        scale = var**0.5

        if scale == 0:
            scale = 1.0

        scales[feature] = scale

    return np.asarray(means), np.asarray(scales)


def benchmark_streaming_scaler(X, size_of_chunk=100, repeat=5):
    """
    This function conducts the comparison between the loop based and
    the vectorised versions of StandardScaler updates.

    Both versions used the same dataset that is processed incrementally
    in fixed-size chunks. The Python-loop implementation of the running
    mean and variance is compared against NumCompute's
    StandardScaler.partial_fit() implementation.

    Then before the benchmark results are reported, the final transformed
    outputs are verified.

    Parameters:
        X:
            Two-dimensional numerical input data.

        size_of_chunk:
            The number of observations that are processed in each
            streaming chunk. The default value is 100.and

        repeat:
            The number of times the each implementation is benchmarked.
            The default value is 5.

    Returns:
        A dict, that contains individual timings, average timing and
        the vectorised speed-up.

    Raises:
        ValueError:
            If the X is not 2D, size_of_chunk is not positive,
            or repeat is not positive.

    Time Complexity:
        O(r * n * f); r is the number of repeats, n is the number
        of observations and f is the number of features.

    Space Complexity:
        O(f + r)
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D array.")

    if size_of_chunk <= 0:
        raise ValueError("size_of_chunk must be greater than 0.")
    if repeat <= 0:
        raise ValueError("repeat must be greater than 0.")

    loop_mean, loop_scale = _loop_streaming_scaler_stats(X, size_of_chunk)

    scaler = numcompute.StandardScaler()

    for begin in range(0, X.shape[0], size_of_chunk):
        chunk = X[begin : begin + size_of_chunk]
        scaler.partial_fit(chunk)

    transformed_loop = (X - loop_mean) / loop_scale
    transformed_vectorised = scaler.transform(X)

    np.testing.assert_allclose(
        transformed_vectorised, transformed_loop, rtol=1e-6, atol=1e-8
    )

    loop_benches = np.zeros(repeat)
    vectorised_benches = np.zeros(repeat)

    for r in range(repeat):
        begin_time = perf_counter()

        _loop_streaming_scaler_stats(X, size_of_chunk)

        end_time = perf_counter()

        loop_benches[r] = (end_time - begin_time) * 1000

        scaler = numcompute.StandardScaler()

        begin_time = perf_counter()

        for begin in range(0, X.shape[0], size_of_chunk):
            chunk = X[begin : begin + size_of_chunk]
            scaler.partial_fit(chunk)

        end_time = perf_counter()

        vectorised_benches[r] = (end_time - begin_time) * 1000

    avg_loop = np.average(loop_benches)
    vec_avg = np.average(vectorised_benches)

    speed_up = avg_loop / vec_avg

    results = {
        "loop_times_ms": loop_benches,
        "vectorised_times_ms": vectorised_benches,
        "loop_average_ms": avg_loop,
        "vectorised_average_ms": vec_avg,
        "speed_up": speed_up,
    }

    print("Streaming StandardScaler Benchmark")
    print(f"The Python loop average was: {avg_loop:.6f} ms")
    print(f"The vectorised average was: {vec_avg:.6f} ms")
    print(f"Vectorised speed-up was: {speed_up:.6f}x")

    return results


def benchmark_buffer_sizes(X, y, buffer_sizes, size_of_chunk=50, repeat=3, max_depth=5):
    """
    This function is to benchmark the DecisionTreeClassifier using
    different streaming buffer sizes.

    To test this fairly, the same dataset is streamed through seperate
    decision trees with each of the supplie buffer sizes. To evaluate
    this, the training runtime is measured across multiple runs, and
    the final tree's accuracy is calculated if the buffer triggers
    a retraining event.

    Parameters:
        X:
            Two-dimensional feature data.

        y:
            One-dimensional array that contains the target
            class labels.

        buffer_sizes:
            Is a sequence of buffer sizes to be compared.

        size_of_chunk:
            Is the number of observations that are supplied to
            every partial_fit() call. The default is 50.

        repeat:
            The number of times benchmarked is repeated for each
            buffer size. The default is 3.

        max_depth:
            The max depth of each decision tree, defaults to 5.

    Returns:
        A list of dicts that contain the buffer size, the individual
        execution times and the average execution times and final
        accuracy.

    Raises:
        ValueError:
            If X and y have different number of observations, there
            are no buffer sizes supplied, or if size_of_chunk or
            repeat values are invalid.

    Time Complexity:
        O(b * r * T); b is the number of buffer sizes to compare,
        r is the number of repeats and T is the cost of streaming
        and retraining one DecisionTreeClassifier across the dataset.

    Space Complexity:
        It is dependant on the DecisionTreeClassifier tree, and
        buffered data. Only one tree is benchmarked at a time.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of observations.")

    if len(buffer_sizes) == 0:
        raise ValueError("buffer_sizes must have at least one value.")

    if size_of_chunk <= 0:
        raise ValueError("size_of_chunk must be greater than 0.")

    if repeat <= 0:
        raise ValueError("repeat must be greater than 0.")

    results = []

    print("Decision Tree Buffer Benchmark")

    for buffer_size in buffer_sizes:
        timings = np.zeros(repeat)
        final_tree = None

        for r in range(repeat):
            tree = numcompute.DecisionTreeClassifier(
                max_depth=max_depth, buffer_size=buffer_size
            )

            begin_time = perf_counter()

            for begin in range(0, X.shape[0], size_of_chunk):
                chunk_X = X[begin : begin + size_of_chunk]
                chunk_y = y[begin : begin + size_of_chunk]

                tree.partial_fit(chunk_X, chunk_y)

            end_time = perf_counter()

            timings[r] = (end_time - begin_time) * 1000

            final_tree = tree

        avg_time = np.average(timings)

        if final_tree.root is not None:
            predictions = final_tree.predict(X)
            accuracy = np.mean(predictions == y)

        else:
            accuracy = None

        result = {
            "buffer_size": buffer_size,
            "times_ms": timings,
            "avg_time_ms": avg_time,
            "accuracy": accuracy,
        }

        results.append(result)

        print(f"Buffer size: {buffer_size}")
        print(f"Average training time: {avg_time:.6f} ms")

        if accuracy is not None:
            print(f"Finaly Accuracy: {accuracy:.4f}")
        else:
            print("Buffer did not trigger retraining, no final accuracy.")

    return results
