
## Overview
*BackboneLearn* is an open-source software package and framework for scaling mixed integer optimization (MIO) problems with indicator variables to high-dimensional problems. This optimization paradigm can naturally be used to formulate fundamental problems in interpretable supervised learning (e.g., sparse regression and decision trees), in unsupervised learning (e.g., clustering), and beyond; *BackboneLearn* solves the aforementioned problems faster than exact methods and with higher accuracy than commonly used heuristics, while also scaling to large problem sizes. The package is built in Python and is user-friendly and easily extendible: users can directly implement a backbone algorithm for their MIO problem at hand.

#### What do we mean by indicators?
Indicators are binary variables that are part of the MIO problem we use to train the aforementioned models.
-  Sparse regression: Each regression coefficient (and the corresponding feature) is paired with an indicator, which is 1 if the coefficient is nonzero and 0 otherwise.
-  Decision trees: An indicator corresponds to a feature in a decision tree node, being nonzero if that feature is chosen for branching at that node.
-  Clustering: An indicator represents whether a pair of data points are in the same cluster, being nonzero if they are clustered together.

## BackboneLearn
The backbone framework, upon which *BackboneLearn* is built, operates in two phases: we first extract a “backbone set” of potentially ``relevant indicators'' (i.e., indicators that are nonzero in the optimal solution) by solving a number of specially chosen, tractable subproblems; we then use traditional techniques to solve a reduced problem to optimality or near-optimality, considering only the backbone indicators. A screening step often proceeds the first phase, to discard indicators that are almost surely irrelevant. For more details, check the paper by Bertsimas and Digalakis Jr (2022) <https://doi.org/10.1007/s10994-021-06123-2>.

### Getting Started
### Installation
Install BackboneLearn using pip:
```python
pip install backbone-learn
```
Note: For ODT implementations, please follow the instructions described in the libary to install compile CBC from source using coinbrew: https://github.com/D3M-Research-Group/odtlearn#cbc-binaries

### Example Usage
Here are some example usages of BackboneLearn for different tasks that you could find in 'notebooks':

#### Sparse Regression with L0BnB Model
```python
from backbone_learn.backbone.backbone_sparse_regression import BackboneSparseRegression
# Initialize BackboneSparseRegression
backbone = BackboneSparseRegression(alpha=0.5, beta=0.5, num_subproblems=5, num_iterations=1, lambda_2=0.001, max_nonzeros=10)
# Fit the model
backbone.fit(X, y)
# Make predictions
y_pred = backbone.predict(X)
```

#### Decision Trees with BendersOCT Model
```python
from backbone_learn.backbone.backbone_decision_tree import BackboneDecisionTree
# Initialize BackboneDecisionTree
backbone = BackboneDecisionTree(alpha=0.5, beta=0.5, num_subproblems=5, num_iterations=1, depth=3, _lambda=0.5)
# Fit the model
backbone.fit(X, y)
# Make predictions
y_pred = backbone.predict(X)
```

#### Clustering with MIO Formulation Model
```python
from backbone_learn.backbone.backbone_clustering import BackboneClustering
# Initialize BackboneClustering
backbone = BackboneClustering(beta=1.0, num_subproblems=5, num_iterations=1, n_clusters=5)
# Fit the model
backbone.fit(X)
# Make predictions
y_pred = backbone.predict(X)
```

### Running Simulations
To run benchmark simulations, please follow these commands and change the hyper-parameters for each script in 'experiments':

```bash
# Install Poetry dependencies
poetry install
# Activate Poetry shell (optional but recommended)
poetry shell
# Run benchmark simulations for decision trees, sparse regression, and clustering
python experiments/benchmark_decision_tree.py
python experiments/benchmark_sparse_regression.py
python experiments/benchmark_clustering.py
```

## Custom Implementation Guide
Follow these steps to implement your backbone algorithms in *BackboneLearn*:
### Custom Screening Method
-  Extend `ScreenSelectorBase` in 'backbone_learn/screen_selector'.
-  Implement the `calculate_utilities` method, which computes utilities (or importances) for each feature based on your screening criteria. Features with the lowest scores may be candidates for elimination. The number of features to keep is defined as a fraction of the total features (`alpha * n_features`).

### Custom Heuristic Method
-  Extend `HeuristicSolverBase` in 'backbone_learn/heuristic_solvers'.
-  Implement `fit` and `get_relevant_features`. The `fit` method trains a model within each subproblem, which needs to be highly efficient. This function fits a sparse model (regression, trees, etc.) to the data inputs of the subproblem. The `get_relevant_features` method identifies and extracts all features deemed relevant by the model. Ensure to add any necessary parameters in the `init` or `fit` method and pass them to the backbone. The fraction of features to include in the feature set for each subproblem is defined by 'beta'. The number of subproblems and iterations are defined by 'num_subproblems' and 'n_iterations', accordingly.

### Custom Exact Solver
-  Extend `ExactSolverBase` in 'backbone_learn/exact_solvers'.
-  Implement `fit` and `predict` methods. The `fit` function is applied to the reduced backbone set and does not need to be efficient; instead, it should use a method with optimality guarantees. The model parameters can again be defined in `init` and passed in the backbone class. The `predict` function ensures that the model can be used for prediction.

### Custom Backbone Algorithm
-  Extend `BackboneSupervised` for supervised algorithms or `BackboneUnsupervised` for unsupervised algorithms in 'backbone_learn/backbone'. The supervised algorithms perform feature selection, while the unsupervised algorithms perform data selection.
-  Implement `set_solvers`. Add your customized screen selector, heuristic solver, and exact solver as class instances. It's optional to define a screen selector and heuristic solver. Parameters for each solver are recommended to be passed manually. See examples in `backbone_learn/backbone/backbone_decision_tree`.

### Example Usage for Customized Backbone Algorithm
Here's an example of how you can crate a custom Backbone algorithm for supervised method:
```python
class CustomBackboneAlgorithm(BackboneSupervised):
    def set_solvers(self, **kwargs):
        # Init screen selector or set None to skip
        self.screen_selector = CustomScreenSelector(**kwarg)
        # Init heuristic solver or set None to skip
        self.heuristic_solver = CustomHeuristicSolver(**kwarg)
         # Init exact solver
        self.exact_solver = CustomExactSolver(**kwarg)
```
Here's how you can use the custom Backbone algorithm:
```python
# Initialize with custom parameters
backbone_algorithm = CustomBackboneAlgorithm(alpha=0.5, beta=0.3, num_subproblems=3,**kwarg)

# Fit the model
backbone_algorithm.fit(X, y)

# Make predictions
predictions = backbone_algorithm.predict(X_new)
```

## Citing BackboneLearn
If you find BackboneLearn useful in your research, please consider citing the following papers.

Paper 1 (Toolkit):

Coming soon

Paper 2 Methodology:

@article{bertsimas2022backbone,
  title={The backbone method for ultra-high dimensional sparse machine learning},
  author={Bertsimas, Dimitris and Digalakis Jr, Vassilis},
  journal={Machine Learning},
  volume={111},
  number={6},
  pages={2161--2212},
  year={2022},
  publisher={Springer}
}

## License
[Apache License 2.0](LICENSE)
