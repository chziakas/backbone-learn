What do we mean by indicators?
==============================

Indicators are binary variables that are part of the MIO problem we use to train the aforementioned models.

- Sparse regression: Each regression coefficient (and the corresponding feature) is paired with an indicator, which is 1 if the coefficient is nonzero and 0 otherwise.
- Decision trees: An indicator corresponds to a feature in a decision tree node, being nonzero if that feature is chosen for branching at that node.
- Clustering: An indicator represents whether a pair of data points are in the same cluster, being nonzero if they are clustered together.
