Decision Trees with BendersOCT Model
------------------------------------

.. code-block:: python

    from backbone_learn.backbone.backbone_decision_tree import BackboneDecisionTree
    # Initialize BackboneDecisionTree
    backbone = BackboneDecisionTree(alpha=0.5, beta=0.5, num_subproblems=5, num_iterations=1, depth=3, _lambda=0.5)
    # Fit the model
    backbone.fit(X, y)
    # Make predictions
    y_pred = backbone.predict(X)
