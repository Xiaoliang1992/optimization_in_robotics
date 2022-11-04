import ipywidgets
from collections import namedtuple
from functools import  partial
from bokeh.plotting import ColumnDataSource
import bokeh.plotting as bkp
from bokeh.models import WheelZoomTool, HoverTool

class FigureLayer:
    def __init__(self):
        self.fig1 = bkp.figure(x_axis_label='x1', y_axis_label='x2', plot_width=600, plot_height=600)
        self.fig2 = bkp.figure(x_axis_label='iter', y_axis_label='g_norm', plot_width=600, plot_height=600)
        self.fig3 = bkp.figure(x_axis_label='iter', y_axis_label='cost', plot_width=600, plot_height=600)

        self.data_source = ColumnDataSource(data={
            'x1': [],
            'x2': [],
            'iter_vec': [],
            'g_norm_vec': [],
            'cost_vec': [],
        })

        f1 = self.fig1.line('x1', 'x2', source=self.data_source, line_width = 2, line_color = 'blue', line_dash = 'solid', legend_label = 'x')
        self.fig1.circle('x1', 'x2', source=self.data_source, size = 10, fill_color="blue")
        hover = HoverTool(renderers=[f1], tooltips=[('x1', '@x1'), ('x2', '@x2')], mode='vline')
        self.fig1.add_tools(hover)

        f2 = self.fig2.line('iter_vec', 'g_norm_vec', source=self.data_source, line_width = 2, line_color = 'blue', line_dash = 'solid', legend_label = 'g_norm(log10)')
        # self.fig2.circle('iter_vec', 'g_norm_vec', source=self.data_source, size = 10, fill_color="blue")
        hover = HoverTool(renderers=[f2], tooltips=[('iter_vec', '@iter_vec'), ('g_norm_vec', '@g_norm_vec')], mode='vline')
        self.fig2.add_tools(hover)

        f3 = self.fig3.line('iter_vec', 'cost_vec', source=self.data_source, line_width = 2, line_color = 'blue', line_dash = 'solid', legend_label = 'cost')
        # self.fig3.circle('iter_vec', 'cost_vec', source=self.data_source, size = 10, fill_color="blue")
        hover = HoverTool(renderers=[f3], tooltips=[('iter_vec', '@iter_vec'), ('cost_vec', '@cost_vec')], mode='vline')
        self.fig3.add_tools(hover)

        self.fig1.toolbar.active_scroll = self.fig1.select_one(WheelZoomTool)
        self.fig2.toolbar.active_scroll = self.fig2.select_one(WheelZoomTool)
        self.fig3.toolbar.active_scroll = self.fig3.select_one(WheelZoomTool)

        self.fig1.legend.click_policy = 'hide'
        self.fig2.legend.click_policy = 'hide'
        self.fig3.legend.click_policy = 'hide'

    def update_data_source(self, data):
        self.data_source.data.update({
            'x1': data['x1'],
            'x2': data['x2'],
            'iter_vec': data['iter_vec'],
            'g_norm_vec': data['g_norm_vec'],
            'cost_vec': data['cost_vec']
        })
class SliderLayer:
    def __init__(self, silder_callback):
        self.solver_type_slider = ipywidgets.IntSlider(layout=ipywidgets.Layout(width='25%'), description= "solver_type",min=0, max=3, value=0, step= 1)
        self.problem_type_slider = ipywidgets.IntSlider(layout=ipywidgets.Layout(width='25%'), description= "problem_type",min=0, max=4, value=0, step= 1)
        self.t0_slider = ipywidgets.FloatSlider(layout=ipywidgets.Layout(width='25%'), description= "t0",min=0.2, max=100.0, value=1.0, step= 0.1)

        ipywidgets.interact(silder_callback, solver_type = self.solver_type_slider, \
                                             problem_type = self.problem_type_slider, \
                                             t0 = self.t0_slider)