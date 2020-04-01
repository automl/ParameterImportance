# 1.1.0

## Major changes
* Add support for SMAC 0.12.0
* Drop support for SMAC < 0.12.0 

# 1.0.7

## Major changes
* Add interactive bokeh-plots for evaluators 

## Interface changes
* Add function `plot_bokeh` to evaluators, returns bokeh-plot

## Minor changes
* Change method to shorten parameter-names on plots
* Add pandas and bokeh to requirements

## Bugfixes
* Support SMAC 0.11.x
* Add traj-alljson format for unambigously readable trajectories
* Fix #112 smac-facade import error
