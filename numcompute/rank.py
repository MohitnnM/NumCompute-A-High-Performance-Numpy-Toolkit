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
