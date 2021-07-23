import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from mpl_toolkits import mplot3d

def plot_obj_f():
    np1 = np.linspace(c1_n.nproc.min(), c1_n.nproc.max(), 50)
    np2 = np.linspace(c2_n.nproc.min(), c2_n.nproc.max(), 50)

    x = np.linspace(48, 50, 30)
    y = np.linspace(48, 54, 30)

    X, Y = np.meshgrid(np1, np2)
    Z = obj_f2([X, Y])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel(c1_n.name)
    ax.set_ylabel(c2_n.name)
    ax.set_zlabel('obj_f')
    ax.plot([561.], [204.], [0.58935], 'o', color='r')

    plt.show()


def obj_f(np):
    np1 = np[0]
    np2 = np[1]
    r = (np1*c1_n.get_fitness(np1) + np2*c2_n.get_fitness(np2)) / (np1 + np2)
    return -r

def obj_f2(np):
    np1 = np[0]
    np2 = np[1]
    r = (np1*c1_n.get_fitness2(np1) + np2*c2_n.get_fitness2(np2)) / (np1 + np2)
    return r

def ineq_constraint1(np):
    np1 = np[0]
    np2 = np[1]
    return - c1_n.get_sypd(np1) + c2_n.get_sypd(np2) + 0.5

def ineq_constraint2(np):
    np1 = np[0]
    np2 = np[1]
    return c1_n.get_sypd(np1) - c2_n.get_sypd(np2) + 0.5

def find_optimal():

    constraint1 = ({'type': 'ineq', 'fun': lambda np: - c1_n.get_sypd(np[0]) + c2_n.get_sypd(np[1]) + 0.5},
                   {'type': 'ineq', 'fun': lambda np: + c1_n.get_sypd(np[0]) - c2_n.get_sypd(np[1]) + 0.5})
    bounds_np1 = (c1_n.nproc.min(), c1_n.nproc.max())
    bounds_np2 = (c2_n.nproc.min(), c2_n.nproc.max())
    bounds = [bounds_np1, bounds_np2]
    ig = np.asarray([500, 500], dtype=np.float32)
    res = minimize(obj_f, ig, method='SLSQP', bounds=bounds, constraints=constraint1, options={'disp': True})
    nproc_c1 = round(res.x[0])
    nproc_c2 = round(res.x[1])
    print("optimal var: x1 = ", nproc_c1, " x2 = ", nproc_c2)

    print("Optimal solution:\n")
    print("%s:\n"
          "nproc: %i\n"
          "SYPD: %.2f\n"
          "Fitness: %.3f\n" % (c1_n.name, nproc_c1, c1_n.get_sypd(nproc_c1), c1_n.get_fitness(nproc_c1)))
    print("%s:\n"
          "nproc: %i\n"
          "SYPD: %.2f\n"
          "Fitness: %.3f\n" % (c2_n.name, nproc_c2, c2_n.get_sypd(nproc_c2), c2_n.get_fitness(nproc_c2)))

    plot_obj_f()
def check_interpo():

    c1.plot_scalability_n()
    c2.plot_scalability_n()

    # To plot the SYPD
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(c1.nproc, c1.sypd)
    ax2.plot(c2.nproc, c2.sypd)

    # To plot the fitness
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    ax3.plot(c1.nproc, c1.compute_fitness())
    ax4.plot(c2.nproc, c2.compute_fitness())

    methods = ['linear', 'slinear', 'quadratic', 'cubic']
    legend = methods.copy()
    legend.insert(0, 'Real')
    for m in methods:
        ### Start using interpolated data
        comp1_new = pd.concat({'nproc': df1.nproc, 'SYPD': df1[m]})
        comp2_new = pd.concat({'nproc': df2.nproc, 'SYPD': df2[m]})

        t1 = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD)
        t2 = Component('NEMO_n', comp2_new.nproc, comp2_new.SYPD)

        ax1.plot(t1.nproc, t1.sypd)
        ax2.plot(t2.nproc, t2.sypd)
        ax3.plot(t1.nproc, t1.compute_fitness())
        ax4.plot(t2.nproc, t2.compute_fitness())


    ax1.set_title("SYPD after interpo for IFS")
    ax1.set_xlabel('nproc')
    ax1.set_ylabel('SYPD')
    ax1.legend(legend)

    ax2.set_title("SYPD after interpo for NEMO")
    ax2.set_xlabel('nproc')
    ax2.set_ylabel('SYPD')
    ax2.legend(legend)

    ax3.set_title("Fitness after interpo for IFS")
    ax3.set_xlabel('nproc')
    ax3.set_ylabel('fitness')
    ax3.legend(legend)

    ax4.set_title("Fitness after interpo for NEMO")
    ax4.set_xlabel('nproc')
    ax4.set_ylabel('fitness')
    ax4.legend(legend)

    plt.show()



def interpolate_data(elpin_cores):
    step = 1
    start = 48
    ## Interpolation
    #methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    methods = ['linear', 'slinear', 'quadratic', 'cubic']
    legend = methods.copy()
    legend.insert(0, 'real')
    xnew = np.arange(start, c1.nproc.max() + 1, step)
    tmp_c1 = pd.Series([c1.sypd[c1.nproc[c1.nproc == n].index[0]] if n in c1.nproc.values
                        else np.NaN for n in xnew])
    df1 = pd.DataFrame({'nproc': xnew, 'real': tmp_c1})
    for m in methods:
        f = interpolate.interp1d(c1.nproc, c1.sypd, kind=m, fill_value="extrapolate")
        ynew = f(xnew)
        df1[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(c1.nproc, c1.sypd, 'o')
        for m in methods:
            plt.plot(xnew, df1[m])
        plt.legend(legend)
        plt.title("Check interpo " + c1.name)
        plt.show()

    ## Interpolation
    elpin_cores = elpin_cores[elpin_cores <= c2.nproc.max()]
    # TODO: Use elpin nproc
    # xnew = pd.Series(elpin_cores)
    xnew = np.arange(start, c2.nproc.max() + 1, step)
    tmp_c2 = pd.Series([c2.sypd[c2.nproc[c2.nproc == n].index[0]] if n in c2.nproc.values
                        else np.NaN for n in xnew])
    df2 = pd.DataFrame({'nproc': xnew, 'real': tmp_c2})
    for m in methods:
        f = interpolate.interp1d(c2.nproc, c2.sypd, kind=m, fill_value="extrapolate")
        ynew = f(xnew)
        df2[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(c2.nproc, c2.sypd, 'o')
        for m in methods:
            plt.plot(xnew, df2[m])
        plt.legend(legend)
        plt.title("Check interpo " + c2.name)
        plt.show()

    return df1, df2


def sypd2chpsy(nproc, sypd):
    return nproc * 24 / sypd


def minmax_rescale(serie):
    return (serie - serie.min()) / (serie.max() - serie.min())


class Component:
    def __init__(self, name, nproc, sypd):
        self.name = name
        self.nproc = nproc
        self.sypd = sypd
        self.chpsy = sypd2chpsy(nproc, sypd)
        self.sypd_n = minmax_rescale(sypd)
        self.chpsy_n = 1 - minmax_rescale(self.chpsy)
        self.fitness = self.compute_fitness()

    def get_sypd(self, nproc):
        return self.sypd[self.nproc == round(nproc)].iloc[0]

    def get_fitness(self, nproc):
        return self.fitness[self.nproc == round(nproc)].iloc[0]

    def get_fitness2(self, nproc):
        nproc_rounded = np.round(nproc)
        r = np.zeros(shape=(nproc_rounded.shape[0], nproc_rounded.shape[1]))
        for i in range(nproc_rounded.shape[0]):
            for j in range(nproc_rounded.shape[1]):
                r[i, j] = self.fitness[self.nproc == nproc_rounded[i, j]]
        return r

    def compute_fitness(self):
        return TTS_r * self.sypd_n + ETS_r * self.chpsy_n

    def plot_sypd(self):
        plt.plot(self.nproc, self.sypd)

    def plot_chpsy(self):
        plt.plot(self.nproc, self.chpsy)

    def plot_scalability(self):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(self.nproc, self.sypd, color='tab:blue')
        ax2.plot(self.nproc, self.chpsy,  color='tab:orange')

        ax1.set_title("Scalability " + self.name)
        ax1.set_xlabel('nproc')
        ax1.set_ylabel('SYPD', color='tab:blue')
        ax2.set_ylabel('CHPSY', color='tab:orange')

        plt.show()

    def plot_scalability_n(self):

        plt.plot(self.nproc, self.sypd_n, color='tab:blue')
        plt.plot(self.nproc, self.chpsy_n, color='tab:orange')
        plt.plot(self.nproc, self.fitness, color='black')

        plt.title("Scalability rescaled for " + self.name)
        plt.xlabel('nproc')
        plt.legend(['TTS', 'ETS', 'TTS ratio = %.1f' % TTS_r + '\n' + 'ETS ratio = %.1f' % ETS_r])

        plt.show()

    def plot_fitness(self):
        plt.plot(self.nproc, self.fitness)


TTS_r = 0.5
ETS_r = 1 - TTS_r
nodesize = 48
method = 'linear'  # ['linear', 'slinear', 'quadratic', 'cubic']

show_plots = True

elpin_cores = pd.Series([48, 92, 144, 192, 229, 285, 331, 380, 411, 476, 521, 563, 605, 665, 694, 759, 806, 826, 905,
                         1008, 1012, 1061, 1129, 1164, 1240, 1275, 1427, 1476, 1632, 1650, 1741, 1870])

comp1 = pd.read_csv("./data/IFS_SR_scalability_ece3.csv")
comp2 = pd.read_csv("./data/NEMO_SR_scalability_ece3.csv")

comp1['CHPSY'] = sypd2chpsy(comp1.nproc, comp1.SYPD)
comp2['CHPSY'] = sypd2chpsy(comp2.nproc, comp2.SYPD)

c1 = Component('IFS', comp1.nproc, comp1.SYPD)
c2 = Component('NEMO', comp2.nproc, comp2.SYPD)

if show_plots:

    c1.plot_fitness()
    c2.plot_fitness()
    plt.title("Fitness")
    plt.legend([c1.name, c2.name])

    plt.show()

df1, df2 = interpolate_data(elpin_cores)
check_interpo()

# TODO: Select one of the methods
comp1_new = pd.DataFrame({'nproc': df1.nproc, 'SYPD': df1['linear']})
comp2_new = pd.DataFrame({'nproc': df2.nproc, 'SYPD': df2['linear']})

c1_n = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD)
c2_n = Component('NEMO_n', comp2_new.nproc, comp2_new.SYPD)

find_optimal()

print("bye")

