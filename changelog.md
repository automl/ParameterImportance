# 1.1.2

## Bugfixes
* Bokeh plots ignored the `show_plot`-argument and always opened browser (#127)

# 1.1.1

## Major changes
* Add support for SMAC 0.12.1 and 0.12.2
* Update args of random-forest to fit latest SMAC-requirements

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
