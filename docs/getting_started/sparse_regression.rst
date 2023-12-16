Sparse Regression with L0BnB Model
----------------------------------

.. code-block:: python

    from backbone_learn.backbone.backbone_sparse_regression import BackboneSparseRegression
    # Initialize BackboneSparseRegression
    backbone = BackboneSparseRegression(alpha=0.5, beta=0.5, num_subproblems=5, num_iterations=1, lambda_2=0.001, max_nonzeros=10)
    # Fit the model
    backbone.fit(X, y)
    # Make predictions
    y_pred = backbone.predict(X)
