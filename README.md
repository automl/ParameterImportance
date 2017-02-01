# PIMP

**PIMP** is an easy to use tool that helps developers to identify the most important parameters of their algorithms.
Given the data of a configuration run with [*SMAC3*](https://github.com/automl/SMAC3), PIMP allows one to use *Forward Selection*, *Efficient Ablation* and *Influence Models* to determine which Parameters have the most influence over the algorithms behaviour.

The documentation can be found [here](https://automl.github.io/ParameterImportance).

Example results of the package look as follows:

## Forward Selection
An Example call of forward-selection:
`
python scripts/evaluate.py --scenario_file scenario.txt --history smac-output/runhistory.json --modus forward-selection
`
Results in an image such as:
![](examples/ForwardSelection.png)


## Surrogate-ablation
An example call of surrogate-ablation:
`
python scripts/evaluate.py --scenario_file scenario.txt --history smac-output/runhistory.json --trajectory smac-output/traj_aclib2.json --modus ablation
`
Results in two plots:
![](examples/Ablationpercentage.png)
![](examples/Ablationperformance.png)
