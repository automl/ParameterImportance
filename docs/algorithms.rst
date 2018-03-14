.. _algos:

Algorithms
==========
.. role:: bash(code)
    :language: bash


In the following we will shortly describe all of the different algorithms that can be used for parameter importance
analysis and are implemented as part of **PyImp**.


.. _ablation:

Ablation
--------

`Ablation <https://link.springer.com/article/10.1007/s10732-014-9275-9>`_ is a local method that determines parameter importances between two given configurations. It thereby looks
which parameter contributed most in a local part of the Configuration Space.
It is an iterative method that changes, in each round, one parameter from the starting configuration to that of the
target configuration. The parameter that resulted in the highest improvement is kept as this rounds most important
parameter. The order determines which parameters are deemed most important and the percentage of improvement tells us
how much influence a parameter has.

In PyImp we implemented an efficient `variant of ablation <http://aad.informatik.uni-freiburg.de/papers/17-AAAI-Surrogate-Ablation.pdf>`_, which replaces costly algorithm runs with cheap to evaluate
surrogates.

.. _forwards:

Forward-Selection
-----------------

Forward-Selection is an iterative method. In each iteration it constructs models that only consider parts of all
available parameters and keeps the one parameter that results in the lowest prediction error for the next round.
The order determines which parameters are deemed most important.

For more details we refer to the original `paper <https://link.springer.com/chapter/10.1007/978-3-642-44973-4_40>`_.

.. _im:

Influence Models
----------------

Influence Models aim to learn a linear model and deems those parameters as most important that result in the highest
weights of the linear model. However it does not necessarily look at all possible parameters, only those that improve
the performance when adding them to the linear model in a forward step. Additionally, it performs one (or more) backwards
steps, in which it checks if parameters have become unimportant due to conditionalities in the Parameter Space.
For more details we refer to the original `paper <https://dl.acm.org/citation.cfm?doid=2786805.2786845>`_.


.. _fa:

fANOVA
------

fANOVA is an efficient parameter importance method, leveraging random forest models fit on the data already gathered by
Bayesian optimization. fANOVA is able to quantify the importance of both single hyperparameters and of interactions
between hyperparameters.

For more details we refer to the original `paper <http://www-devel.cs.ubc.ca/~hoos/Publ/HutEtAl14b.pdf>`_.


.. _lp:

LPI
---

*L*ocal *P*arameter *I*mportance ist the most local parameter importance analysis method.
It is inspired by the human strategy to look
for further improved parameter configurations or to understand the importance
of parameter changes in the neighborhood of a parameter configuration. For ex-
ample, most users are interested in understanding which parameters in optimized
parameter configurations are crucial for the achieved performance.
Using an EPM, we study performance changes of a configuration along each
parameter. To quantify the importance of a parameter value, we compute
the variance of all cost values by changing the parameter and then compute the fraction
of all variances.