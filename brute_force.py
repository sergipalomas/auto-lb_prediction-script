import pandas as pd
import numpy as np
import sys
from component_class import sypd2chpsy
import matplotlib.pyplot as plt
from matplotlib import cm
from component_class import Component
from cplx_opt import plot_obj_f


def calc_fitness(sypd, chpsy, TTS_r, ETS_r):
    return TTS_r * sypd + ETS_r * chpsy


def obj_f2(nproc, c1_n, c2_n):
    nproc1 = nproc[0]
    nproc2 = nproc[1]
    r = np.vectorize(c1_n.get_fitness)(nproc1) + np.vectorize(c2_n.get_fitness)(nproc2)
    return r


def plot_obj(c1_n, c2_n):
    nproc1_n = np.linspace(c1_n.nproc.min(), c1_n.nproc.max(), 50).round()
    nproc2_n = np.linspace(c2_n.nproc.min(), c2_n.nproc.max(), 50).round()

    X, Y = np.meshgrid(nproc1_n, nproc2_n)
    Z = obj_f2([X, Y], c1_n, c2_n)

    for ii in range(0, 91, 30):
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel(c1_n.name)
        ax.set_ylabel(c2_n.name)
        ax.set_zlabel('obj_f')
        ax.view_init(elev=20., azim=ii)
        plt.show()

def brute_force(c1_n, c2_n):

    # Sort before matching
    c1_n_sort = c1_n.sypd.sort_values('SYPD')
    c2_n_sort = c2_n.sypd.sort_values('SYPD')

    # For each combination nproc/SYPD of c1_n, give the nproc of c2_n which minimizes the differences in the SYPD between
    # both components
    closest = pd.merge_asof(c1_n_sort, c2_n_sort, on='SYPD', direction='nearest')
    closest['f1'] = closest.apply(lambda x: c1_n.get_fitness(x['nproc_x']), axis=1)
    closest['f2'] = closest.apply(lambda x: c2_n.get_fitness(x['nproc_y']), axis=1)
    closest['objective_f'] = closest.f1 + closest.f2
    optimal_result = closest.iloc[closest.objective_f.idxmax()]
    plt.plot(closest.SYPD, closest.objective_f)
    plt.plot(optimal_result.SYPD, optimal_result.objective_f, 'o')
    plt.title("Objective function")
    plt.show()

    plot_obj(c1_n, c2_n)

    optimal_result = closest.iloc[closest.objective_f.idxmax()]

    return optimal_result
