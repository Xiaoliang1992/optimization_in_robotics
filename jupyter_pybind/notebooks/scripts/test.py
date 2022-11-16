from bokeh.plotting import figure, output_file, show, ColumnDataSource
import ipywidgets
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import layout, column, row
from IPython.core.display import display, HTML
from bokeh.models import WheelZoomTool, HoverTool
import sys, os

sys.path.append('../../..')
from build import solver_py

output_notebook()
# output_file("log_lines.html")

# data

data_source = ColumnDataSource(data={
    'x1': [0],
    'x2': [0]
})

fig1 = figure(x_axis_label='x', y_axis_label='y', plot_width=400, plot_height=400)

f2 = fig1.line('x1', 'x2', source=data_source, line_width = 2, line_color = 'red', line_dash = 'solid', legend_label = 'y=x')
f1 = fig1.circle('x1', 'x2', source=data_source, radius = 10, fill_color="red")
hover = HoverTool(renderers=[f1], tooltips=[('x1', '@x1'), ('x2', '@x2')], mode='vline')
fig1.add_tools(hover)
fig1.toolbar.active_scroll = fig1.select_one(WheelZoomTool)
fig1.legend.click_policy = 'hide'


# 显示结果
show(fig1)
