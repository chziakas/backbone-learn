# Contributing to BackboneLearn

## Introduction

Thank you for considering a contribution to BackboneLearn!
This project is dedicated to advancing the field of mixed integer optimization in machine learning. Contributions in various forms, such as new methods, bug fixes, or documentation enhancements, are all highly valued.

## How to Contribute

### Step 1: Understanding the Project

Explore the [BackboneLearn documentation](Readme.md) to understand the project's scope, functionalities, and architecture. We encourage future contributors to review our research paper and other key open-source libraries.

### Step 2: Setting Up

**Fork and Clone the Repository:**

*   Fork the BackboneLearn repository on GitHub.
*   Clone your fork to your local environment.

**Environment Setup:**

*   Install Python 3.9, if not already installed.
*   Set up your development environment using Poetry for dependency management:

```bash
   pip install poetry
   poetry install
```
*   Utilize pre-commit hooks to maintain code quality:

```bash
   pre-commit install
```

### Step 3: Making Changes

**Finding or Creating an Issue:**

*   Check the [GitHub issues](link-to-issues) for existing tasks or bugs.
*   You can also create new issues for proposing features or improvements.

**Creating a Branch:**

*   Create a new branch in your fork for your changes.

**Developing:**

*   Implement your changes, ensuring to adhere to the project's coding standards.
*   Write or update tests using Pytest.
*   Run and pass all tests before submission.

### Step 4: Submitting Your Contribution

**Committing and Pushing Changes:**

*   Commit your changes with clear messages.
*   Push your changes to your fork on GitHub.

**Creating a Pull Request:**

*   Open a pull request against the main BackboneLearn repository.
*   Describe your changes in detail and link to any relevant issues.

**Code Review and GitHub Actions:**

*   Engage in the code review process.
*   Your code will be automatically checked by GitHub Actions.
*   Make necessary changes based on feedback.

## Reporting Issues and Suggestions

**Check Existing Issues:**

Before reporting, search the GitHub issue to ensure it's unique.

**Create New Issue:**

Navigate to 'Issues' in the BackboneLearn repository and click 'New Issue'.

**Fill the Template and Submit:**

Provide a detailed title and description. Include steps to reproduce, expected and actual behavior, code snippets, or screenshots if applicable. After completing the form, click 'Submit new issue'.

## Custom Implementation Guide

To build customized backbone algorithms for BackboneLearn, follow these steps:

### Custom Screening Method

**Extend `ScreenSelectorBase` in `'backbone_learn/screen_selector'`.**

**Implement the `calculate_utilities` Method:**

*   Compute utilities or importances for each feature based on your criteria.
*   Features with the lowest scores may be considered for elimination.
*   The number of features to keep is `alpha * n_features`.

### Custom Heuristic Method

**Extend `HeuristicSolverBase` in `'backbone_learn/heuristic_solvers'`.**
**Implement `fit` and `get_relevant_features`:**

*   Train a model within each subproblem efficiently.
*   Fit a sparse model to the data inputs of the subproblem.
*   Identify and extract relevant features.
*   Define necessary parameters in the `init` or `fit` method.

### Custom Exact Solver

**Extend `ExactSolverBase` in `'backbone_learn/exact_solvers'`.**
**Implement `fit` and `predict` Methods:**

*   Apply `fit` to the reduced backbone set.
*   Use a method with optimality guarantees.
*   Ensure the model can be used for prediction.

### Custom Backbone Algorithm

**Extend `BackboneSupervised` or `BackboneUnsupervised` in `'backbone_learn/backbone'`.**
**Implement `set_solvers`:**

*   Add customized screen selector, heuristic solver, and exact solver.
*   Optionally define a screen selector and heuristic solver.
*   Pass parameters manually.

### Example Usage for Customized Backbone Algorithm

```python
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

```

## Questions or Issues?

If you encounter any issues or have questions, please open a discussion issue on GitHub.
