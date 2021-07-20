import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        plt.plot(self.nproc, self.sypd, self.nproc, self.chpsy)
    def plot_chpsy(self):
        plt.plot(self.nproc, self.chpsy)
    def plot_fitness(self):
        plt.plot(self.nproc, self.fitness)


TTS_r = 0.8
ETS_r = 1 - TTS_r

comp1 = pd.read_csv("./data/IFS_SR_scalability_ece3.csv")
comp2 = pd.read_csv("./data/NEMO_SR_scalability_ece3.csv")

comp1['CHPSY'] = sypd2chpsy(comp1.nproc, comp1.SYPD)
comp2['CHPSY'] = sypd2chpsy(comp2.nproc, comp2.SYPD)

c1 = Component('IFS', comp1.nproc, comp1.SYPD)
c2 = Component('NEMO', comp2.nproc, comp2.SYPD)

c1.plot_sypd()
c2.plot_sypd()
plt.title("Scalability")
plt.legend([c1.name, c2.name])
plt.show()

c1.plot_chpsy()
c2.plot_chpsy()
plt.title("Scalability Cost")
plt.legend([c1.name, c2.name])
#plt.show()

c1.plot_fitness()
c2.plot_fitness()
plt.title("Fitness")
plt.legend([c1.name, c2.name])
#plt.show()

x = np.arange(48, 1008, 48)

df = pd.DataFrame()