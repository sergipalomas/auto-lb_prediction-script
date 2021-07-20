import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def interpolate_data(elpin_cores):
    ## Interpolation
    #methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    methods = ['linear', 'slinear', 'quadratic', 'cubic']
    legend = methods.copy()
    legend.insert(0, 'real')
    xnew = np.arange(48, c1.nproc.max() + nodesize, nodesize)
    tmp_c1 = pd.Series([c1.sypd[c1.nproc[c1.nproc == n].index[0]] if n in c1.nproc.values
                        else np.NaN for n in xnew])
    df1 = pd.DataFrame({'nproc': xnew, 'real': tmp_c1})
    plt.plot(c1.nproc, c1.sypd, 'o')
    for m in methods:
        f = interpolate.interp1d(c1.nproc, c1.sypd, kind=m)
        ynew = f(xnew)
        df1[m] = pd.DataFrame({m: ynew})
        plt.plot(xnew, ynew)
    plt.legend(legend)
    plt.title("Check interpo " + c1.name)
    plt.show()

    ## Interpolation
    elpin_cores = elpin_cores[elpin_cores <= c2.nproc.max()]
    xnew = pd.Series(elpin_cores)
    tmp_c2 = pd.Series([c2.sypd[c2.nproc[c2.nproc == n].index[0]] if n in c2.nproc.values
                        else np.NaN for n in xnew])
    df2 = pd.DataFrame({'nproc': xnew, 'real': tmp_c2})
    plt.plot(c2.nproc, c2.sypd, 'o')
    for m in methods:
        f = interpolate.interp1d(c2.nproc, c2.sypd, kind=m)
        ynew = f(xnew)
        df2[m] = pd.DataFrame({m: ynew})
        plt.plot(xnew, ynew)
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
        self.fitness = (TTS_r * self.sypd_n + ETS_r * self.chpsy_n)

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
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax1.plot(self.nproc, self.sypd_n, color='tab:blue')
        ax2.plot(self.nproc, self.chpsy_n, color='tab:orange')
        ax3.plot(self.nproc, self.fitness, color='black')

        ax1.set_title("Scalability reescaled for " + self.name)
        ax1.set_xlabel('nproc')
        ax1.set_ylabel('SYPD_norm', color='tab:blue')
        ax2.set_ylabel('CHPSY_norm', color='tab:orange')
        ax3.legend(['TTS ratio = %.1f' % TTS_r + '\n' + 'ETS ratio = %.1f' % ETS_r])

        plt.show()

    def plot_fitness(self):
        plt.plot(self.nproc, self.fitness)

TTS_r = 0.5
ETS_r = 1 - TTS_r
nodesize = 48
show_plots = False

elpin_cores = pd.Series([48, 92, 144, 192, 229, 285, 331, 380, 411, 476, 521, 563, 605, 665, 694, 759, 806, 826, 905,
                         1008, 1012, 1061, 1129, 1164, 1240, 1275, 1427, 1476, 1632, 1650, 1741, 1870])

comp1 = pd.read_csv("./data/IFS_SR_scalability_ece3.csv")
comp2 = pd.read_csv("./data/NEMO_SR_scalability_ece3.csv")

comp1['CHPSY'] = sypd2chpsy(comp1.nproc, comp1.SYPD)
comp2['CHPSY'] = sypd2chpsy(comp2.nproc, comp2.SYPD)

c1 = Component('IFS', comp1.nproc, comp1.SYPD)
c2 = Component('NEMO', comp2.nproc, comp2.SYPD)

if show_plots:
    c1.plot_scalability()
    c1.plot_scalability_n()

    c2.plot_scalability()
    c2.plot_scalability_n()


    c1.plot_fitness()
    c2.plot_fitness()
    plt.title("Fitness")
    plt.legend([c1.name, c2.name])

    plt.show()

df1, df2 = interpolate_data(elpin_cores)

### Start using interpolated data
comp1_new = pd.concat({'nproc': df1.nproc, 'SYPD': df1.method})

