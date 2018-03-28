Manual
======
.. role:: bash(code)
    :language: bash


In the following we will show how to use **PyImp**.

Before proceeding with this Quick-Start guide, you sould make sure you have all requirenment installed (see :doc:`installation`).
PyImp is developed for **python3.5** and above!

.. _quick:

Quick Start
-----------

To show you how easy it is to use *PyImp* we will do it by example.
In the examples folder you'll find the folder spear_qcp.

First navigate to this folder.

.. code-block:: bash

    cd examples/spear_qcp

It contains all the necessary files, to optimize *Spear* using *SMAC3*.
To see a detailed description of what each file is used for, we refer to *SMACs* `manual <https://automl.github.io/SMAC3/stable/manual.html#spear-qcp>`_. All the files are provided so you can run SMAC for yourself to create new output to use with *PyImp*

The folder *smac-output* contains the *runhistory* and *trajectory* files of a 10 minute
SMAC run for the specified scenario.

To use **PyImp** simply execute the following call in the spear_qcp folder:

.. code-block:: bash

    pimp --scenario_file scenario.txt --history './*/runhistory.json' --modus forward-selection

With this call, PyImp will read in the scenarios info and all runhistories in this folder and evaluate the parameter importances,
using forward selection (see :doc:`algorithms`).

PyImp will create a json file that contains the names and importance values of all parameters that the analysis algorithm
checked. You can easily load this file using pythons json module.

Further, every algorithm will create one ore more plots in the same directory in which the evaluate script was called.


This is all you need to have in order to determine the parameter importance of your algorithm.

Using PyImp in python
---------------------
TODO

.. _opts:

Usage
_____

.. code-block:: bash

    usage: pimp [-h] -S SCENARIO_FILE -M
            {ablation,forward-selection,influence-model,all,fanova,lpi,incneighbor}
            [{ablation,forward-selection,influence-model,all,fanova,lpi,incneighbor} ...]
            -H HISTORY [--seed SEED] [-V {INFO,DEBUG}] [-T TRAJECTORY]
            [-N NUM_PARAMS] [-P MAX_SAMPLE_SIZE] [-I] [-C] [-F OUT_FOLDER]
            [-D WDIR] [--fanova_cut_at_default] [--fanova_no_pairs]
            [--incneigh_quantify_perf_improvement] [--forward_sel_feat_imp]
            [--marginalize_over_instances]

    optional arguments:
      -h, --help            show this help message and exit

    Required Options:
      -S SCENARIO_FILE, --scenario_file SCENARIO_FILE
                            scenario file in AClib format (default: None)
      -M {ablation,forward-selection,influence-model,all,fanova,lpi,incneighbor} [{ablation,forward-selection,influence-model,all,fanova,lpi,incneighbor} ...], --modus {ablation,forward-selection,influence-model,all,fanova,lpi,incneighbor} [{ablation,forward-selection,influence-model,all,fanova,lpi,incneighbor} ...]
                            Analysis method(s) to use (default: None)
      -H HISTORY, --history HISTORY
                            runhistory file (default: None)

    Optional Options:
      --seed SEED           random seed (default: 12345)
      -V {INFO,DEBUG}, --verbose_level {INFO,DEBUG}
                            verbosity (default: 20)
      -T TRAJECTORY, --trajectory TRAJECTORY
                            Path to trajectory file (default: None)
      -N NUM_PARAMS, --num_params NUM_PARAMS
                            Number of parameters to evaluate (default: 0)
      -P MAX_SAMPLE_SIZE, --max_sample_size MAX_SAMPLE_SIZE
                            Number of samples from runhistorie(s) used. -1 -> use
                            all (default: -1)
      -I, --impute          Impute censored data (default: False)
      -C, --table           Save result table (default: False)
      -F OUT_FOLDER, --out-folder OUT_FOLDER
                            Folder to store results in (default: None)
      -D WDIR, --working_dir WDIR
                            Directory to load all folders from. (default: .)
      --fanova_cut_at_default
                            Cut fANOVA results at the default. This quantifies
                            importance only in terms of improvement over the
                            default. (default: False)
      --fanova_no_pairs     fANOVA won't compute pairwise marginals (default:
                            True)
      --incneigh_quantify_perf_improvement
                            incumbent neighborhood computes importance via
                            performance improvement (default: True)
      --forward_sel_feat_imp
                            forward selection for feature importance (default:
                            False)
      --marginalize_over_instances
                            Deactivate preprocessing step in which instances are
                            marginalized away to speedup ablation, forward-
                            selection and incumbent neighborhood predictions
                            (default: False)

The trajectory file is only needed for **local analysis methods** such as *ablation* and *LPI*.

num_params determines how many parameters have to be evaluated. It is useful if you are only interested in the *n* most
important parameters. If it is not set or below 1 or above the maximum number of parameters to evaluate, it will evaluate
all possible parameters.
