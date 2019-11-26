import os

import numpy as np
from bokeh.io import output_file, save, show
from bokeh.models import BasicTickFormatter, Row
from bokeh.plotting import figure


def bokeh_boxplot(labels, mean, std, x_label, y_label, runtime, inc_indices):
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

def bokeh_line_uncertainty(values, mean, std, x_log, x_label, y_label, inc_indices):
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
    # Save and show...
    if plot_name:
        outdir = os.path.dirname(plot_name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not plot_name.endswith('.html'):
            plot_name = plot_name + '.html'
        output_file(plot_name)
        save(layout)
    if True or show_plot:
        show(layout)
