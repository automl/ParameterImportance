# ParameterImportance

## Forward Selection
An Example call of forward-selection:
`
python scripts/evaluate.py --scenario_file scenario.txt --history smac-output/runhistory.json --trajectory smac-output/traj_aclib2.json --modus forward-selection
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