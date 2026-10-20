"""
Pipeline infrastructure for composing the NumCompute workflows.

This module supports the minimal Transformer and Estimator protocols that
are used by NumCompute, and provides a Pipeline class that chains multiple
individual processing steps in a sequence. The transformer components 
follow the fit/transform interface, while estimator components 
follow a fit/predict interface.

The aim of this is to allow the output of one transformation to be passed
to the next, which provides NumCompute with a consistent way to construct
repeatable preprocessing and numerical workflows. There is no implementation
of machine learning models in this module.
"""

class Pipeline:
    """
        Chains the multiple individual processing steps into a sequential
        workflow.

        Every step, sans the last one, must follow the transformer interface
        by providing the callable methods, fit() and transform(). The last
        step could either be a transformer with methods fit() and transform()
        or an estimator with methods fit() and predict().

        Every step, sans the last, that is classfied as a transformer is fitted,
        and then its transformed output is passed along to the next step. 

        Parameters:
            steps:
                A sequence of (name, step) pairs. Ensure that step names are 
                unique strings. Every step before the last must provide fit()
                and transform() and the last step must provide fit() and 
                either transform() or predict()

        Attributes:
            steps:
                The ordered sequence of pipeline steps.

            fitted:
                A boolean variable to check if fit() has been successfully
                called.   
    """

    def __init__(self, steps):
        """
            The initialisation of the pipeline and validating the 
            configuration of steps.

            Parameters:
                steps:
                    A sequence of (name, step) pairs defining what order the
                    components are applied in. Ensure that step names are 
                    unique strings.

            Raises:
                TypeError:
                    If steps are not in the expected (name, step) form, or
                    if a step name is not a string.

                ValueError:
                    If no steps are provided, if a step contain more or less
                    than 2 elements, or if duplicate step names were found.

                AttributeError:
                    If any of the steps, excluding the last, do not provide 
                    callable fit() or transform() methods, or if the last 
                    step does not provide a callable fit() method and 
                    either a transform() or a predict().
        """

        self.steps = steps
        self.fitted = False
        self._validate_steps()


    def _validate_steps(self):
        """
            Validates the interface and structure of all pipeline steps.

            Every step, excluding the last, must provide callable fit()
            and transform() methods, and the last step must provide a 
            callable fit() method and either a transform() or a predict().

            Parameters:
                None
            
            Returns:
                None

            Raises:
                TypeError:
                    If steps are not in the expected (name, step) form, or
                    if a step name is not a string.

                ValueError:
                    If no steps are provided, if a step contain more or less
                    than 2 elements, or if duplicate step names were found.

                AttributeError:
                    If any of the steps, excluding the last, do not provide 
                    callable fit() or transform() methods, or if the last 
                    step does not provide a callable fit() method and 
                    either a transform() or a predict().

            Time Complexity:
                O(s), where s is the number of steps in pipeline. 

            Space Complexity:
                O(s), because of sorting and checking all step names.
        """

        names = []
        for item in self.steps:
            if not isinstance(item, (tuple, list)):
                raise TypeError(
                    f"Expected a tuple or a list pair of (name, step) "
                    f"but got: {type(item).__name__}."
                )

            if len(item) != 2:
                raise ValueError(
                    f"Expected a pair of 2 elements, but got "
                    f"a sequence of length: {len(item)}."
                )

            name, step = item

            if not isinstance(name, str):
                raise TypeError(
                    f"Expected name to be str, but got: "
                    f"{type(name).__name__}."
            )

            names.append(name)
        
        if len(self.steps) == 0:
            raise ValueError("Pipeline steps cannot be empty")
        

        if not len(names) == len(set(names)):
            raise ValueError(
                "Step names are expected to be unique, but "
                f"duplicates were found."
            )

        for name, step in self.steps[:-1]:
            if not callable(getattr(step, "fit", None)):
                raise AttributeError(
                    f"Step {name!r} missing callable fit() method."
                )

            if not callable(getattr(step, "transform", None)):
                raise AttributeError(
                    f"Step {name!r} missing callable transform() method."
                )

        name, last_step = self.steps[-1]

        if not callable(getattr(last_step, "fit", None)):
            raise AttributeError(
                f"Last step {name!r} missing callable fit() method."
            )

        if (
            not callable(getattr(last_step, "transform", None)) 
            and not callable(getattr(last_step, "predict", None))
        ):
            raise AttributeError(
                f"Last step {name!r} missing callable transform()"
                " or predict() method."
            )


    def fit(self, X):
        """
            Fits each pipeline step sequentially on X.

            Each intermidiate transformer is fitted on the current 
            representation of X, then transformed before being 
            passed on to the next step. Then the last step is 
            fitted on the output of all the prior transformations.

            Parameters:
                X: 
                    Input data passed through the pipeline.

            Returns: 
                Self: 
                    The instance of the fitted Pipeline
            
            Time Complexity:
                Is dependant on the combined fit() and transform() complexity
                of pipeline steps.
            
            Space Complexity:
                Is dependant on the transformed array produced by the pipeline
                steps.
        """

        curr_X = X

        for _, step in self.steps[:-1]:
            step.fit(curr_X)
            curr_X = step.transform(curr_X)

        _, last_step = self.steps[-1]
        
        last_step.fit(curr_X)

        self.fitted = True

        return self
        

    def transform(self, X):
        """
            Transforms each pipeline step sequentially on X.

            The transform() method of each step is applied in pipeline
            order, with output of a step, becoming the input of the 
            next step.

            Parameters:
                X:
                    The data to transform

            Returns:
                The tranformed output produced by the last pipeline step.

            Raises:
                RuntimeError:
                    If the pipeline has not yet been fitted
                
                AttributeError:
                    If the last step is not a tranformer but an estimator.

            Time Complexity:
                Is dependant on the combined transform() complexity
                of all pipeline steps.
            
            Space Complexity:
                Is dependant on the transformed array produced by the pipeline
                steps.
        """

        if not self.fitted:
            raise RuntimeError("Pipeline has not yet been fitted.")

        _, last_step = self.steps[-1]     

        if not callable(getattr(last_step, "transform", None)):
            raise AttributeError(
                "Last step missing callable transform() method."
            )

        curr_X = X

        for _, step in self.steps:
            curr_X = step.transform(curr_X)

        return curr_X

    def fit_transform(self, X):
        """
            Combines previous 2 steps, and fits the pipeline to X,
            then returns the transformed result. 

            Parameters:
                X:
                    The input data to be fitted and transformed.

            Returns:
                The tranformed output produced by the last pipeline step.
            
            Raises:
                AttributeError:
                    If the last step does not have a transform() methods.
            
            Time Complexity:
                Is dependant on the combined cost of fit() and transform()
                across all pipeline steps.
            
            Space Complexity:
                Is dependant on the transformed array produced during fitting
                and transformation.
        """
        _, last_step = self.steps[-1]     

        if not callable(getattr(last_step, "transform", None)):
            raise AttributeError(
                "Last step missing callable transform() method."
            )

        return self.fit(X).transform(X)

    def predict(self, X):
        """
            Generated predictions using the fitted pipeline.

            Excluding the final estimator, all steps transform X 
            sequentially. With the transformed data being passed onto 
            the last steps predict() method.

            Parameters:
                X: 
                    The data that requires the predictions.
            
            Returns:    
                Predictions produced by the last estimator.

            Raises:
                RuntimeError:
                    if the pipeline has not yet been fitted.

                AttributeError:
                    If the last step does not have a predict() methods.

            Time Complexity:
                Depends on the combined transform() complexity of all steps,
                exluding the last, and the predict() complexity of the final
                estimator.

            Space Complexity:
                Depends on the transformed arrays and prediction output.
        """
        
        if not self.fitted:
            raise RuntimeError("Pipeline has not yet been fitted.")

        _, last_step = self.steps[-1]     

        if not callable(getattr(last_step, "predict", None)):
            raise AttributeError(
                f"Last step missing callable predict() method."
            )

        curr_X = X

        for _, step in self.steps[:-1]:
            curr_X = step.transform(curr_X)

        return last_step.predict(curr_X)