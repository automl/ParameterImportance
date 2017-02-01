.. _algos:

Algorithms
==========
.. role:: bash(code)
    :language: bash


In the following we will shortly describe all of the different algorithms that can be used for parameter importance
analysis and are implemented as part of **PIMP**.


.. _ablation:

Ablation
--------

TODO

.. _forwards:

Forward-Selection
-----------------

TODO

.. _im:

Influence Models
----------------

Influence Models aim to learn a linear model and deems those parameters as most important that result in the highest
weights of the linear model. However it does not necessarily look at all possible parameters, only those that improve
the performance when adding them to the linear model in a forward step. Additionally, it performs one (or more) backwards
steps, in which it checks if parameters have become unimportant due to conditionalities in the Parameter Space.
For more details we refer to the original `paper <https://dl.acm.org/citation.cfm?doid=2786805.2786845>`_.
