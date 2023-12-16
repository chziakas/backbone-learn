Custom Implementation Guide
---------------------------

To build customized backbone algorithms for BackboneLearn, follow these steps:

Custom Screening Method
^^^^^^^^^^^^^^^^^^^^^^^

**Extend `ScreenSelectorBase` in `'backbone_learn/screen_selector'`.**

**Implement the `calculate_utilities` Method:**

- Compute utilities or importances for each feature based on your criteria.
- Features with the lowest scores may be considered for elimination.
- The number of features to keep is `alpha * n_features`.

Custom Heuristic Method
^^^^^^^^^^^^^^^^^^^^^^^

**Extend `HeuristicSolverBase` in `'backbone_learn/heuristic_solvers'`.**
**Implement `fit` and `get_relevant_features`:**

- Train a model within each subproblem efficiently.
- Fit a sparse model to the data inputs of the subproblem.
- Identify and extract relevant features.
- Define necessary parameters in the `init` or `fit` method.

Custom Exact Solver
^^^^^^^^^^^^^^^^^^^

**Extend `ExactSolverBase` in `'backbone_learn/exact_solvers'`.**
**Implement `fit` and `predict` Methods:**

- Apply `fit` to the reduced backbone set.
- Use a method with optimality guarantees.
- Ensure the model can be used for prediction.

Custom Backbone Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^

**Extend `BackboneSupervised` or `BackboneUnsupervised` in `'backbone_learn/backbone'`.**
**Implement `set_solvers`:**

- Add customized screen selector, heuristic solver, and exact solver.
- Optionally define a screen selector and heuristic solver.
- Pass parameters manually.

Example Usage for Customized Backbone Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class CustomBackboneAlgorithm(BackboneSupervised):
        def set_solvers(self, **kwargs):
            self.screen_selector = CustomScreenSelector(**kwargs)
            self.heuristic_solver = CustomHeuristicSolver(**kwargs)
            self.exact_solver = CustomExactSolver(**kwargs)

    # Initialize with custom parameters
    backbone_algorithm = CustomBackboneAlgorithm(alpha=0.5, beta=0.3, num_subproblems=3, **kwargs)

    # Fit the model
    backbone_algorithm.fit(X, y)

    # Make predictions
    predictions = backbone_algorithm.predict(X_new)
