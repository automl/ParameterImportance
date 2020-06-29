import os

import numpy as np
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.models import BasicTickFormatter, BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import Inferno256, Category20
from bokeh.plotting import figure
from bokeh.transform import transform


def bokeh_heatmap_num(data, x_label, y_label, x_log, y_log, grid_column='zz'):
    """
    Plot heatmap for numerical values (resolution predefined by dataframe)
    Grid-values are the values being filled in the headt-map rectangles

    Parameters
    ----------
    data: pandas.Dataframe
        dataframe
    x_label, y_label: str
        labels for axes
    x_log, y_log: bool
        whether to plot axes in (natural) log
    grid_column: str
        name of column with grid data in dataframe
    """
    # Reshape into 1D array
    df = pd.DataFrame(data.stack(), columns=[grid_column]).reset_index()
    source = ColumnDataSource(df)
    mapper = LinearColorMapper(palette=Inferno256, low=df[grid_column].min(), high=df[grid_column].max())
    x_range = (data.index[0], data.index[-1])
    y_range = (data.columns[0], data.columns[-1])
    return _bokeh_heatmap(source, mapper, x_label, y_label, x_range, y_range, x_log=x_log, y_log=y_log)

def bokeh_heatmap_cat(data, x_label, y_label, grid_column='zz'):
    """
    Plot heatmap for categorical values (resolution defined by number of choices)
    Grid-values are the values being filled in the headt-map rectangles

    Parameters
    ----------
    data: pandas.Dataframe
        dataframe
    x_label, y_label: str
        labels for axes
    grid_column: str
        name of column with grid data in dataframe
    """
    # Reshape into 1D array
    df = pd.DataFrame(data.stack(), columns=[grid_column]).reset_index()
    source = ColumnDataSource(df)
    mapper = LinearColorMapper(palette=Inferno256, low=df[grid_column].min(), high=df[grid_column].max())
    x_range = [str(c) for c in data.index]
    y_range = [str(c) for c in reversed(data.columns)]
    return _bokeh_heatmap(source, mapper, x_label, y_label, x_range, y_range)

def _bokeh_heatmap(source, colormapper, x_label, y_label, x_range, y_range, x_log=False, y_log=False):
    p = figure(x_range=x_range, y_range=y_range,
               x_axis_type="log" if x_log else "linear",
               y_axis_type="log" if y_log else "linear",
               toolbar_location=None, tools="")
    p.rect(x=x_label, y=y_label, width=1, height=1, source=source,
           fill_color=transform('zz', colormapper),
           line_color=None)
    color_bar = ColorBar(color_mapper=colormapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=20),
                         formatter=BasicTickFormatter(use_scientific=False))
    p.add_layout(color_bar, 'right')
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    return p

def bokeh_boxplot(labels, mean, std, x_label, y_label, runtime, inc_indices):
    """
    Plot a boxplot in bokeh

    Parameters
    ----------
    labels: List[str]
        labels for the x-axis (for the different boxes)
    mean: List[float]
        means, same length as labels
    std: List[float]
        stds, same length as labels
    x_label, y_label: str
        axis-labels
    runtime: bool
        whether runtime is analyzed, if so will limit y-axis to 0 for std
    inc_indices: List[int]
        list of indices in values that are to be marked as incumbents

    Returns
    -------
    plot: bokeh.plotting.figure
        the bokeh figure
    """
    p = figure(x_range=labels, height=350, width=800)
    upper = mean + std
    lower = mean - std
    if runtime:
        lower = [0 if l < 0 else l for l in lower]
    p.vbar(labels, 0.7, mean, upper, fill_color="#E08E79", line_color="black")
    p.vbar(labels, 0.7, mean, lower, fill_color="#3B8686", line_color="black")
    if len(inc_indices) > 0:
        p.circle(list([labels[idx] for idx in inc_indices]), list([float(mean[idx]) for idx in inc_indices]),
                 name='incumbent', color='black')

    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.formatter = BasicTickFormatter(use_scientific=False)

    return p

def bokeh_multiline(data, x_limits, y_limits, x_column, line_columns, y_label, z_label):
    """
    Bokeh multiline plot

    Parameters
    ----------
    data: pandas.DataFrame
        data to be plotted
    x_limits, y_limits: Tuple(float, float)
        limits for x and y
    x_column: str
        name of the column in dataframe to be plotted on x-axis
    line_columns: List[str]
        names of the columns to be plotted as lines
    y_label: str
        label for y-axis
    z_label: str
        name of the categorical parameter (for tooltips)

    Returns
    -------
    plot: bokeh.plotting.figure
        the bokeh figure
    """
    ys = [data[lc] for lc in line_columns]
    xs = [data[x_column] for _ in range((len(ys)))]
    palette = Category20[len(xs)] if len(xs) > 2 else ['blue', 'red'] if len(xs) == 2 else ['blue']
    source = ColumnDataSource(dict(
        xs=xs,
        ys=ys,
        color=palette,
        label=line_columns,
    ))

    p = figure(x_range=x_limits,
               y_range=y_limits,
               toolbar_location=None, tools="")

    p.multi_line(xs="xs", ys="ys", color="color",
                 legend="label",
                 source=source)
    p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
        (z_label, '@label')
    ]))

    p.xaxis.axis_label = x_column
    p.yaxis.axis_label = y_label

    return p

def bokeh_line_uncertainty(values, mean, std, x_log, x_label, y_label, inc_indices):
    """
    Return a bokeh plot with a simple line plot and uncertainty marked as filled area above and below plot.

    Parameters
    ----------
    values: List
        values for x-axis (same length as mean and std)
    mean: List[float]
        means (same length as values and std)
    std: List[float]
        uncertainty to be plotted around the mean (same length as values and mean)
    x_log: bool
        plot x-axis in log
    x_label, y_label: str
        axis-labels
    inc_indices: List[int]
        list of indices in values that are to be marked as incumbents

    Returns
    -------
    plot: bokeh.plotting.figure
        the bokeh figure
    """
    lower_curve = mean - std
    upper_curve = mean + std

    p = figure(x_range=(values[0], values[-1]), x_axis_type="log" if x_log else "linear", height=350, width=800)
    p.line(values, mean)
    band_x = np.append(values, values[::-1])
    band_y = np.append(lower_curve, upper_curve[::-1])
    p.patch(band_x, band_y, color='#7570B3', fill_alpha=0.2)

    if len(inc_indices) > 0:
        p.circle(list([values[idx] for idx in inc_indices]), list([float(mean[idx]) for idx in inc_indices]),
                 name='incumbent', color='black')

    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.formatter = BasicTickFormatter(use_scientific=False)

    return p

def save_and_show(plot_name, show_plot, layout):
    """
    Save `layout` in `plot_name` (which has to be the complete path including the filename, '.html' will be appended if
    necessary) and show plot in default browser if `show_plot` is `True`.
    """
    if plot_name:
        outdir = os.path.dirname(plot_name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not plot_name.endswith('.html'):
            plot_name = plot_name + '.html'
        output_file(plot_name)
        save(layout)
    if show_plot:
        show(layout)

def shorten_unique(names, keep_first=4, keep_last=4):
    """
    Shorten strings, inserting '(...)', while keeping them unique.

    Parameters
    ----------
    names: List[str]
        list of strings to be shortened
    keep_first: int
        always keep the first N letters
    keep_last: int
        always keep the last N letters

    Returns
    -------
    shortened_names: List[str]
        list with shortened strings
    """
    short, cut_chars, longest_str = [], 0, max([len(p) for p in names])
    while len(set(short)) != len(set(names)) and cut_chars <= longest_str:
        short = [p[:keep_first + cut_chars] + '(...)' + p[-keep_last:]
                 if len(p) > sum([keep_first, keep_last, cut_chars])
                 else p for p in names]
        cut_chars += 1
    return short
