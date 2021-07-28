import matplotlib.pyplot as plt
import numpy as np


def sypd2chpsy(nproc, sypd):
    return nproc * 24 / sypd


def minmax_rescale(series):
    return (series - series.min()) / (series.max() - series.min())


class Component:
    def __init__(self, name, nproc, sypd, TTS_r, ETS_r):
        self.name = name
        self.nproc = nproc
        self.sypd = sypd
        self.max_sypd = max(sypd)
        self.chpsy = sypd2chpsy(nproc, sypd)
        self.sypd_n = minmax_rescale(sypd)
        self.chpsy_n = 1 - minmax_rescale(self.chpsy)
        self.TTS_r = TTS_r
        self.ETS_r = ETS_r
        self.fitness = self.compute_fitness()


    def get_nproc_from_sypd(self, sypd):
        return self.nproc[self.sypd == round(sypd)].iloc[0]

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
        return self.TTS_r * self.sypd_n + self.ETS_r * self.chpsy_n

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
        plt.legend(['TTS', 'ETS', 'TTS ratio = %.1f' % self.TTS_r + '\n' + 'ETS ratio = %.1f' % self.ETS_r])

        plt.show()

    def plot_fitness(self):
        plt.plot(self.nproc, self.fitness)
