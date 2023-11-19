
## Overview



## backbone_learn



## backbone_learn

## Get started

### Installation
ToDo: Define method
```bash
pip install backbone-learn
```

You could find some example notebooks.

Note: For ODT implementations, please follow the instructions described in the libary to install compile CBC from source using coinbrew: https://github.com/D3M-Research-Group/odtlearn#cbc-binaries

### Run simulations
```poetry install
poetry shell
python experiments/sparse_regression_script.py
python experiments/clustering_script.py
python experiments/decision_trees_script.py
```


## Add your own implementation
This guide explains how to create your own Backbone algoritms. You would need to
1. Implement your own screening method by implementing calculate_utilities() function from
1. Implement your heuristic solver by implementing fit() and get_relevant_features() functions
2. Implement your exact sovler by implementing fit() and predict()



1. Implement a Custom Screening Method

First, create a subclass of ScreenSelectorBase and implement the calculate_utilities method. This method should compute utilities (or importances) for each feature based on your screening criteria.

```python
class CustomScreenSelector(ScreenSelectorBase):
    def calculate_utilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray::
        # Implement logic here to calculate feature utilities
```


2. Implement a Custom Heuristic Method

Next, create a subclass of HeuristicSolverBase and implement the fit, model, and get_relevant_features methods. This solver fits a model to the data using a heuristic approach and identifies relevant features.

```python
class CustomHeuristicSolver(HeuristicSolverBase):
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Implement logic here to fit the heuristic model
        # Store the fitted model in self._model


    def get_relevant_variables(self, **kwargs)->
        # Logic to identify and return relevant features
```

3. Implement a Custom Exact Solver

Create a subclass of ExactSolverBase and implement the fit, model, and predict methods. This solver fits a model using an exact or more rigorous approach and makes predictions.

```python
class CustomExactSolver(ExactSolverBase):
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Logic to fit the exact model
        # Store the fitted model in self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Logic to make predictions using the fitted model
```
4. Build Your Custom Backbone Algorithm

To create your own Backbone algorithm, you can inherit from the BackboneBase class and define your custom solvers. The set_solvers method allows you to specify the screen selector, exact solver, and heuristic solver.

Here's an example of how you can assemble a custom Backbone algorithm:

```python
class CustomBackboneAlgorithm(BackboneBase):
    def set_solvers(self, **kwargs):
        # Extract parameters specific to each solver
        parameter_1 = kwargs.get('parameter_1')
        parameter_2 = kwargs.get('parameter_2')

        # Initialize custom solvers with extracted parameters
        self.screen_selector = CustomScreenSelector()
        self.exact_solver = CustomExactSolver(parameter_1)
        self.heuristic_solver = CustomHeuristicSolver(parameter_2)

        # You coud always skip screen_selector and exact_solver by defining them None
        self.screen_selector = None
        self.exact_solver = None
```


In this example, CustomBackboneAlgorithm automatically inherits the fit and predict methods from BackboneBase. The set_solvers method is used to initialize the specific screen selector, exact solver, and heuristic solver. This method extracts the relevant parameters from kwargs which are passed during object initialization.


5. Example Usage
Here's how you can use the custom Backbone algorithm:

```python
# Initialize with custom parameters
backbone_algorithm = CustomBackboneAlgorithm(alpha=0.5, beta=0.3, num_subproblems=3,...)

# Fit the model
backbone_algorithm.fit(X, y)

# Make predictions
predictions = backbone_algorithm.predict(X_new)
```
## License

[Apache License 2.0](LICENSE)
