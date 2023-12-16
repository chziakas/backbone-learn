Backbone Framework
------------------

The backbone framework, upon which *BackboneLearn* is built, operates in two phases: we first extract a “backbone set” of potentially ``relevant indicators`` (i.e., indicators that are nonzero in the optimal solution) by solving a number of specially chosen, tractable subproblems; we then use traditional techniques to solve a reduced problem to optimality or near-optimality, considering only the backbone indicators. A screening step often precedes the first phase, to discard indicators that are almost surely irrelevant. For more details, check the paper by Bertsimas and Digalakis Jr (2022) `here <https://doi.org/10.1007/s10994-021-06123-2>`_.
