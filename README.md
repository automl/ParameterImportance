# PyImp

**PyImp** is now on pypi.
To install it simply run
`
pip install pyimp
`
If you want to run fANOVA with PyImp you will have to manually install it via
`
pip install git+http://github.com/automl/fanova.git@master
`

**PyImp** is an easy to use tool that helps developers to identify the most important parameters of their algorithms.
Given the data of a configuration run with [*SMAC3*](https://github.com/automl/SMAC3), PyImp allows one to use *Forward Selection*, *Efficient Ablation* and *Influence Models* to determine which Parameters have the most influence over the algorithms behaviour.

PyImp can be used with argcomplete. To enable autocompletion of PyImp
arguments, add the following line to your .bashrc or .profile:
`
eval "$(register-python-argcomplete pyimp)"
`

The documentation can be found [here](https://automl.github.io/ParameterImportance).

Example results of the package look as follows:

## Forward Selection
An Example call of forward-selection:
`
pimp --scenario_file scenario.txt --history smac-output/runhistory.json --modus forward-selection
`
Results in an image such as:
![](examples/ForwardSelection.png)


## Surrogate-ablation
An example call of surrogate-ablation:
`
pimp --scenario_file scenario.txt --history smac-output/runhistory.json --trajectory smac-output/traj_aclib2.json --modus ablation
`
Results in two plots:
![](examples/Ablationpercentage.png)
![](examples/Ablationperformance.png)

