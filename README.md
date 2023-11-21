
## Overview

We present *BackboneLearn*: an open-source software package and framework for scaling mixed integer optimization (MIO) problems with indicator variables to high-dimensional problems. 
*BackboneLearn* solves fundamental problems in interpretable supervised learning including sparse regression and decision trees, in unsupervised learning including clustering, and beyond, all of which can naturally be formulated using the aforementioned optimization paradigm. As we have extensively documented in prior work, the backbone framework can handle supervised learning problems with millions of features in minutes and our *BackboneLearn* implementation indeed accelerates the solution of exact methods while improving upon the quality of commonly used heuristics. The package is built in Python and is user-friendly and easily extendible: users can directly implement a backbone algorithm for their MIO problem at hand.

#### What do we mean by indicators?

Indicators are binary variables that are part of the MIO problem we use to train the aforementioned models. 
- In the context of sparse regression, where require that only a few regression coefficients are nonzero, we couple each regression coefficient (and the corresponding feature) with an inidicator. The indicator is 1 if the coefficient is indeed nonzero, and 0 otherwise. 
- In the context of decision trees, the indicator corresponding to feature j in split node t is nonzero iff, at node t, we branch on feature j. 
- In the context of clustering, the indicator corresponding to a pair of data points (i,j) is nonzero iff i and j are put in the same cluster.

## backbone_learn

The backbone framework, upon which *BackboneLearn* is built, operates in two phases: we first extract a “backbone set” of potentially ``relevant indicators'' (i.e., indicators that are nonzero in the optimal solution) by solving a number of specially chosen, tractable subproblems; we then use traditional techniques to solve a reduced problem to optimality or near-optimality, considering only the backbone indicators. A screening step often proceeds the first phase, to discard indicators that are almost surely irrelevant. 


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
python experiments/benchmark_decision_tree.py
python experiments/benchmark_sparse_regression.py
python experiments/benchmark_clustering.py
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
