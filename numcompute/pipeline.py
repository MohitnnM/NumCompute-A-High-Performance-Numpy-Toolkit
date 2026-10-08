"""
Pipeline infrastructure for composing the NumCompute workflows.

This module defines the minimal Transformer and Estimator protocols that
are used by NumCompute, and provides a Pipeline class that chains multiple
individual processing steps in a sequence. The transformer components 
follow the fit/transform interface, while estimator components 
follow a fit/predict interface.

The aim of this is to allow the output of one transformation to be passed
to the next, which provides NumCompute with a consistent way to construct
repeatable preprocessing and numerical workflows. There is no implementation
of machine learning models in this module.
"""
