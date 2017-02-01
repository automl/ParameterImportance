Manual
======
.. role:: bash(code)
    :language: bash


In the following we will show how to use **PIMP**.

Before proceeding with this Quick-Start guide, you sould make sure you have all requirenment installed (see :doc:`installation`).
PIMP is developed for **python3.5** and above!

.. _quick:

Quick Start
-----------

To show you how easy it is to use *PIMP* we will do it by example.
In the examples folder you'll find the folder spear_qcp.

First navigate to this folder.

.. code-block:: bash

    cd examples/spear_qcp

It contains all the necessary files, to optimize *Spear* using *SMAC3*.
To see a detailed description of what each file is used for, we refer to *SMACs* `manual <https://automl.github.io/SMAC3/stable/manual.html#spear-qcp>`_. All the files are provided so you can run SMAC for yourself to create new output to use with *PIMP*

The folder *smac-output* contains the *runhistory* and *trajectory* files of a 10 minute
SMAC run for the specified scenario.

To use **PIMP** simply execute the following call in the spear_qcp folder:

.. code-block:: bash

    python ~/git/importance/scripts/evaluate.py --scenario_file scenario.txt --history './*/runhistory.json' --modus forward-selection

With this call, PIMP will read in the scenarios info and all runhistories in this folder and evaluate the parameter importances,
using forward selection (see :doc:`algorithms`).

PIMP will create a json file that contains the names and importance values of all parameters that the analysis algorithm
checked. You can easily load this file using pythons json module.

Further, every algorithm will create one ore more plots in the same directory in which the evaluate script was called.


This is all you need to have in order to determine the parameter importance of your algorithm.

.. _opts:

Usage
_____

.. code-block:: bash

    usage: evaluate.py [-h] --scenario_file SCENARIO_FILE --modus
                   {ablation,forward-selection,influence-model} --history
                   HISTORY [--seed SEED] [--verbose_level {INFO,DEBUG}]
                   [--trajectory TRAJECTORY] [--num_params NUM_PARAMS]
    optional arguments:
      -h, --help            show this help message and exit
    Required Options:
      --scenario_file SCENARIO_FILE
                            scenario file in AClib format (default: None)
      --modus {ablation,forward-selection,influence-model}
                            Analysis method to use (default: None)
      --history HISTORY     runhistory file (default: None)
    Optional Options:
      --seed SEED           random seed (default: 12345)
      --verbose_level {INFO,DEBUG}
                            verbosity (default: 20)
      --trajectory TRAJECTORY
                            Path to trajectory file (default: None)
      --num_params NUM_PARAMS
                            Number of parameters to evaluate (default: 0)

The trajectory file is only needed for **local analysis methods** such as *ablation*.

num_params determines how many parameters have to be evaluated. It is useful if you are only interested in the *n* most
important parameters. If it is not set or below 1 or above the maximum number of parameters to evaluate, it will evaluate
all possible parameters.
