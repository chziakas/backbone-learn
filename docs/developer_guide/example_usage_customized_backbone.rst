Example Usage for Customized Backbone Algorithm
-----------------------------------------------

Here's an example of how you can create a custom Backbone algorithm for a supervised method:

.. code-block:: python

    class CustomBackboneAlgorithm(BackboneSupervised):
        def set_solvers(self, **kwargs):
            # Init screen selector or set None to skip
            self.screen_selector = CustomScreenSelector(**kwargs)
            # Init heuristic solver or set None to skip
            self.heuristic_solver = CustomHeuristicSolver(**kwargs)
            # Init exact solver
            self.exact_solver = CustomExactSolver(**kwargs)

Here's how you can use the custom Backbone algorithm:

.. code-block:: python

    # Initialize with custom parameters
    backbone_algorithm = CustomBackboneAlgorithm(alpha=0.5, beta=0.3, num_subproblems=3, **kwargs)

    # Fit the model
    backbone_algorithm.fit(X, y)

    # Make predictions
    predictions = backbone_algorithm.predict(X_new)
