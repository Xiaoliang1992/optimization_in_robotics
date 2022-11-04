import sys, os
sys.path.append('..')
from lib.figure_layer import *

sys.path.append('../../..')
from build import solver_py
import numpy as np

import ipywidgets
from bokeh.io import output_notebook, push_notebook, output_file
from bokeh.layouts import layout, column, row
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important;  }</style>"))
output_notebook()

fig = FigureLayer()

solver = solver_py.Solver()
solver_param = solver_py.SolverParameters()
x0 = solver_py.EigenClass().GetVectorXd(2)

solver.SetSolver(0)
solver.SetProblem(1)


def silder_callback(solver_type, problem_type, t0):
    kwargs = locals()
    solver_param.t0 = t0
    solver.SetSolver(solver_type)
    solver.SetProblem(problem_type)
    solver.SetParam(solver_param)
    x1 = solver.Solve(x0)
    info = solver.GetInfo()

    print('solution:')
    print(x1)

    data = {}

    x1 =[]
    x2 =[]
    iter_vec = []
    g_norm_vec = []
    cost_vec = []

    for i in range(len(info.x_vec)):
        x1.append(info.x_vec[i][0])
        x2.append(info.x_vec[i][1])
        iter_vec.append(info.iter_vec[i])
        g_norm_vec.append(info.g_norm_vec[i])
        cost_vec.append(info.cost_vec[i])

    data['x1'] = x1
    data['x2'] = x2
    data['iter_vec'] = iter_vec
    data['g_norm_vec'] = g_norm_vec
    data['cost_vec'] = cost_vec

    fig.update_data_source(data)

    push_notebook()

bkp.show(row(fig.fig1, fig.fig2, fig.fig3), notebook_handle=True)
slider_class = SliderLayer(silder_callback)


