import matplotlib.pyplot as plt
import pandas as pd


def sypd2chsy(nproc, sypd):
    return nproc * 24 / sypd


def minmax_rescale(series):
    if series.shape[0] > 1:
        return (series - series.min()) / (series.max() - series.min())
    else:
        return 1


class Component:
    def __init__(self, name, nproc, sypd, nproc_restriction, ts_info, ts_nproc, TTS_r, ETS_r):
        self.name = name
        self.nproc = nproc
        self.min_nproc = min(nproc)
        self.sypd = pd.DataFrame({'nproc': nproc, 'SYPD': sypd})
        self.max_sypd = max(sypd)
        self.chsy = pd.DataFrame({'nproc': nproc, 'CHPSY': sypd2chsy(nproc, sypd)})
        self.speedup = pd.DataFrame({'nproc': nproc, 'speedup': self.get_speedup(nproc)})
        self.efficiency = pd.DataFrame({'nproc': nproc, 'efficiency': self.get_efficiency(nproc)})
        self.sypd_n = pd.DataFrame({'nproc': nproc, 'SYPD': minmax_rescale(self.sypd.SYPD)})
        self.chsy_n = pd.DataFrame({'nproc': nproc, 'CHPSY': 1 - minmax_rescale(self.chsy.CHPSY)})
        self.TTS_r = TTS_r
        self.ETS_r = ETS_r
        self.fitness = pd.DataFrame({'nproc': nproc, 'fitness': self.compute_fitness()})
        self.fitness2 = pd.DataFrame({'nproc': nproc, 'fitness2': self.compute_fitness2()})
        self.nproc_restriction = pd.Series(nproc_restriction)
        self.ts_info = ts_info
        self.ts_nproc = ts_nproc


    def get_nproc_from_sypd(self, sypd):
        return self.nproc[self.sypd.SYPD == round(sypd)].iloc[0]

    def get_sypd(self, nproc):
        return self.sypd[self.nproc == nproc].SYPD.iloc[0]

    def get_sypd_v(self, nproc):
        return self.sypd[self.nproc.isin(nproc)].SYPD

    def get_sypd_n(self, nproc):
        return self.sypd_n[self.sypd_n.nproc == nproc].SYPD

    def get_speedup(self, nproc):
        return self.get_sypd_v(nproc) / self.get_sypd(self.min_nproc)

    def get_efficiency(self, nproc):
        return self.get_speedup(nproc) / (nproc/self.min_nproc)

    def get_chsy(self, nproc):
        return self.chsy[self.nproc == nproc].CHPSY.iloc[0]

    def get_chsy2(self, nproc):
        return self.chsy[self.nproc.isin(nproc)].CHPSY

    def get_chsy_n(self, nproc):
        return self.chsy_n[self.chsy_n.nproc == nproc].CHPSY

    def get_fitness(self, nproc):
        return self.fitness[self.nproc.isin(nproc)]

    def compute_fitness(self):
        return self.TTS_r * self.sypd_n.SYPD + self.ETS_r * self.chsy_n.CHPSY

    def compute_fitness2(self):
        return self.speedup.speedup * self.efficiency.efficiency

    def plot_sypd(self):
        plt.plot(self.nproc, self.sypd)

    def plot_chsy(self):
        plt.plot(self.nproc, self.chsy)

    def plot_scalability(self, *opt_nproc):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        if len(opt_nproc) == 1:
            optimal_nproc = opt_nproc[0]
            optimal_sypd = self.get_sypd(opt_nproc)
            optimal_chsy = self.get_chsy(opt_nproc)
            ax1.axvline(x=opt_nproc, ls='-.', c='k', label='Optimal: %i proc' % optimal_nproc)
            self.sypd.plot(x="nproc", y="SYPD", color='tab:blue', ax=ax1, label='SYPD: %.2f' % optimal_sypd, legend=False)
            self.chsy.plot(x="nproc", y="CHPSY", color='tab:orange', ax=ax2, label='CHPSY: %i' % optimal_chsy,legend=False)
            sypd_text = ' %.2f' % optimal_sypd
            ax1.plot(opt_nproc[0], optimal_sypd, 'ko', markersize=5)
            ax1.text(opt_nproc[0], optimal_sypd, sypd_text)
            chsy_text = ' %i' % optimal_chsy
            ax2.plot(opt_nproc[0], optimal_chsy, 'ko', markersize=5)
            ax2.text(opt_nproc[0], optimal_chsy, chsy_text)
            lns = [ax1.lines[0], ax1.lines[1], ax2.lines[0]]
            labels = [l.get_label() for l in lns]
            ax1.legend(handles=lns, labels=labels, loc=0)
            ax1.set_title("Solution for " + self.name)

        else:
            self.sypd.plot(x="nproc", y="SYPD", color='tab:blue', ax=ax1, legend=False)
            self.chsy.plot(x="nproc", y="CHPSY", color='tab:orange', ax=ax2, legend=False)
            lns = [ax1.lines[1], ax2.lines[0]]
            labels = [l.get_label() for l in lns]
            ax1.legend(handles=lns, labels=labels, loc=0)
            ax1.set_title("Solution for " + self.name)

        ax1.set_xlabel('nproc')
        ax1.set_ylabel('SYPD', color='tab:blue')
        ax2.set_ylabel('CHPSY', color='tab:orange')

        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        fig_name = self.name + "_solution.png"
        plt.savefig("./img/" + fig_name)
        #plt.show()

    def plot_scalability_n(self, *opt_nproc):
        fig, ax1 = plt.subplots()
        self.sypd_n.plot(x="nproc", y="SYPD", color='tab:blue', ax=ax1)
        self.chsy_n.plot(x="nproc", y="CHPSY", color='tab:orange', ax=ax1)
        self.fitness.plot(x="nproc", y="fitness", color='black', ax=ax1)

        if len(opt_nproc) == 1:
            optimal_nproc = opt_nproc[0]
            optimal_sypd_n = self.get_sypd_n(optimal_nproc)
            optimal_chsy_n = self.get_chsy_n(optimal_nproc)
            optimal_fitness = self.get_fitness([optimal_nproc]).fitness
            ax1.axvline(x=optimal_nproc, ls='-.', c='k', label='Optimal: %i proc' % optimal_nproc, alpha=1.)
            sypd_text = ' %.2f' % optimal_sypd_n
            ax1.plot(optimal_nproc, optimal_sypd_n, 'ko', markersize=5)
            ax1.text(optimal_nproc, optimal_sypd_n, sypd_text)
            chsy_text = ' %.2f' % optimal_chsy_n
            ax1.plot(optimal_nproc, optimal_chsy_n, 'ko', markersize=5)
            ax1.text(optimal_nproc, optimal_chsy_n, chsy_text)
            fitness_text = ' %.2f' % optimal_fitness
            ax1.plot(optimal_nproc, optimal_fitness, 'ko', markersize=5)
            ax1.text(optimal_nproc, optimal_fitness, fitness_text)

        plt.xlabel('nproc')
        ax1.set_title("Scalability rescaled and optimal value for " + self.name + "\n" + "TTS: %.2f    ETS: %.2f" %
                      (self.TTS_r, self.ETS_r))
        plt.legend(['TTS ratio', 'ETS ratio', 'Fitness'], loc=0)

        fig_name = self.name + "_scalability_normalized.png"
        plt.savefig("./img/" + fig_name)
        #plt.show()

    def plot_fitness(self, *opt_nproc):
        fig, ax1 = plt.subplots()
        self.fitness.plot(x="nproc", y="fitness", color='tab:blue', ax=ax1, legend=False)
        if len(opt_nproc) == 1:
            optimal_nproc = opt_nproc[0]
            ax1.plot(optimal_nproc, self.get_fitness([opt_nproc]).fitness, 'ro', markersize=5)
            optimal_fitness = self.get_fitness(optimal_nproc)
            fitness_text = ' %.2f' % optimal_fitness
            ax1.text(optimal_nproc, optimal_fitness, fitness_text)

        plot_title = "Fitness for component " + self.name
        ax1.set_title(plot_title)
        ax1.set_ylabel("Fitness")
