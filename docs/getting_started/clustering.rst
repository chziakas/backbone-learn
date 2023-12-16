Clustering with MIO Formulation Model
-------------------------------------

.. code-block:: python

    from backbone_learn.backbone.backbone_clustering import BackboneClustering
    # Initialize BackboneClustering
    backbone = BackboneClustering(beta=1.0, num_subproblems=5, num_iterations=1, n_clusters=5)
    # Fit the model
    backbone.fit(X)
    # Make predictions
    y_pred = backbone.predict(X)
