import matplotlib.pyplot as plt
import pandas as pd


def sypd2chpsy(nproc, sypd):
    return nproc * 24 / sypd


def minmax_rescale(series):
    return (series - series.min()) / (series.max() - series.min())


class Component:
    def __init__(self, name, nproc, sypd, TTS_r, ETS_r):
        self.name = name
        self.nproc = nproc
        self.sypd = pd.DataFrame({'nproc': nproc, 'SYPD': sypd})
        self.max_sypd = max(sypd)
        self.chpsy = pd.DataFrame({'nproc': nproc, 'CHPSY': sypd2chpsy(nproc, sypd)})
        self.sypd_n = pd.DataFrame({'nproc': nproc, 'SYPD': minmax_rescale(self.sypd.SYPD)})
        self.chpsy_n = pd.DataFrame({'nproc': nproc, 'CHPSY': 1 - minmax_rescale(self.chpsy.CHPSY)})
        self.TTS_r = TTS_r
        self.ETS_r = ETS_r
        self.fitness = pd.DataFrame({'nproc': nproc, 'fitness': self.compute_fitness()})


    def get_nproc_from_sypd(self, sypd):
        return self.nproc[self.sypd.SYPD == round(sypd)].iloc[0]

    def get_sypd(self, nproc):
        return self.sypd[self.nproc == nproc].SYPD.iloc[0]

    def get_fitness(self, nproc):
        return self.fitness[self.nproc == nproc].fitness.iloc[0]

    def get_fitness2(self, nproc):
        return self.fitness[self.nproc.isin(nproc)]

    def compute_fitness(self):
        return self.TTS_r * self.sypd_n.SYPD + self.ETS_r * self.chpsy_n.CHPSY

    def plot_sypd(self):
        plt.plot(self.nproc, self.sypd)

    def plot_chpsy(self):
        plt.plot(self.nproc, self.chpsy)

    def plot_scalability(self):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        self.sypd.plot(x="nproc", y="SYPD", color='tab:blue', ax=ax1, legend=False)
        self.chpsy.plot(x="nproc", y="CHPSY", color='tab:orange', ax=ax2, legend=False)

        ax1.legend(loc=(0.05, 0.9))
        ax2.legend(loc=(0.05, 0.83))
        ax1.set_title("Scalability " + self.name)
        ax1.set_xlabel('nproc')
        ax1.set_ylabel('SYPD', color='tab:blue')
        ax2.set_ylabel('CHPSY', color='tab:orange')

        ax1.set_ylim(ymin=0)
        ax2.set_ylim(ymin=0)

        plt.show()

    def plot_scalability_n(self):
        fig, ax1 = plt.subplots()
        self.sypd_n.plot(x="nproc", y="SYPD", color='tab:blue', ax=ax1)
        self.chpsy_n.plot(x="nproc", y="CHPSY", color='tab:orange', ax=ax1)
        self.fitness.plot(x="nproc", y="fitness", color='black', ax=ax1)

        plt.title("Scalability rescaled for " + self.name)
        plt.xlabel('nproc')
        plt.legend(['TTS', 'ETS', 'TTS ratio = %.1f' % self.TTS_r + '\n' + 'ETS ratio = %.1f' % self.ETS_r])

        plt.show()

    def plot_fitness(self):
        plt.plot(self.nproc, self.fitness)
