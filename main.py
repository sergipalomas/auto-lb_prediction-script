import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from cplx_opt import find_optimal
from mpl_toolkits import mplot3d


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

        t1 = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD, TTS_r, ETS_r)
        t2 = Component('NEMO_n', comp2_new.nproc, comp2_new.SYPD, TTS_r, ETS_r)

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
    tmp_c1 = pd.Series([c1.sypd.SYPD[c1.nproc[c1.nproc == n].index[0]] if n in c1.nproc.values
                        else np.NaN for n in xnew])
    df1 = pd.DataFrame({'nproc': xnew, 'real': tmp_c1})
    for m in methods:
        f = interpolate.interp1d(c1.nproc, c1.sypd.SYPD, kind=m, fill_value="extrapolate")
        ynew = f(xnew).round(2)
        df1[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(c1.nproc, c1.sypd.SYPD, 'o')
        for m in methods:
            plt.plot(xnew, df1[m])
        plt.legend(legend)
        plt.title("Check interpo " + c1.name)
        plt.show()

    ## Interpolation
    # TODO: Create a loop for each component
    elpin_cores = elpin_cores[elpin_cores <= c2.nproc.max()]
    # TODO: Use elpin nproc
    # xnew = pd.Series(elpin_cores)
    xnew = np.arange(start, c2.nproc.max() + 1, step)
    tmp_c2 = pd.Series([c2.sypd.SYPD[c2.nproc[c2.nproc == n].index[0]] if n in c2.nproc.values
                        else np.NaN for n in xnew])
    df2 = pd.DataFrame({'nproc': xnew, 'real': tmp_c2})
    for m in methods:
        f = interpolate.interp1d(c2.nproc, c2.sypd.SYPD, kind=m, fill_value="extrapolate")
        ynew = f(xnew).round(2)
        df2[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(c2.nproc, c2.sypd.SYPD, 'o')
        for m in methods:
            plt.plot(xnew, df2[m])
        plt.legend(legend)
        plt.title("Check interpo " + c2.name)
        plt.show()

    return df1, df2


def print_result(c1_n, c2_n, optimal_result):
    print("Optimal for TTS=%.1f, ETS=%.1f: \n" % (TTS_r, ETS_r))
    print("Number of processes for %s: %.2f" % (c1_n.name, optimal_result.nproc_x))
    print("Number of processes for %s: %.2f" % (c2_n.name, optimal_result.nproc_y))

    print("Fitness %s: %.2f" % (c1_n.name, optimal_result.f1))
    print("Fitness %s: %.2f" % (c2_n.name, optimal_result.f2))
    print("Objective function: %f" % optimal_result.objective_f)

    print("Expected coupled SYPD: %.2f" % optimal_result.SYPD)
    print("Expected coupled CHPSY: %i" % (c1_n.get_chpsy(optimal_result.nproc_x) + c2_n.get_chpsy(optimal_result.nproc_y)))
    print("%s CHPSY: %i" % (c1_n.name, c1_n.get_chpsy(optimal_result.nproc_x)))
    print("%s CHPSY: %i" % (c2_n.name, c2_n.get_chpsy(optimal_result.nproc_y)))

    plt.plot(c1_n.nproc, c1_n.fitness.fitness)
    plt.plot(c2_n.nproc, c2_n.fitness.fitness)
    plt.plot(optimal_result.nproc_x, c1_n.get_fitness(optimal_result.nproc_x), 'o')
    plt.plot(optimal_result.nproc_y, c2_n.get_fitness(optimal_result.nproc_y), 'o')
    plt.title("Fitness value")
    plt.legend(['IFS', 'NEMO'])
    plt.show()

    c1_n.plot_scalability()
    c2_n.plot_scalability()


if __name__ == "__main__":
    # Some Parameters
    TTS_r = 0.5
    ETS_r = 1 - TTS_r
    nodesize = 48
    method = 'cubic'  # ['linear', 'slinear', 'quadratic', 'cubic']
    max_nproc = 1750

    show_plots = False

    elpin_cores = pd.Series([48, 92, 144, 192, 229, 285, 331, 380, 411, 476, 521, 563, 605, 665, 694, 759, 806, 826, 905,
                             1008, 1012, 1061, 1129, 1164, 1240, 1275, 1427, 1476, 1632, 1650, 1741, 1870])

    comp1 = pd.read_csv("./data/IFS_SR_scalability_ece3.csv")
    comp2 = pd.read_csv("./data/NEMO_SR_scalability_ece3.csv")

    from component_class import Component

    c1 = Component('IFS', comp1.nproc, comp1.SYPD, TTS_r, ETS_r)
    c2 = Component('NEMO', comp2.nproc, comp2.SYPD, TTS_r, ETS_r)

    # Interpolate data
    df1, df2 = interpolate_data(elpin_cores)

    if show_plots:

        c1.plot_fitness()
        c2.plot_fitness()
        plt.title("Fitness")
        plt.legend([c1.name, c2.name])

        plt.show()

        check_interpo()

    # TODO: Select one of the methods
    comp1_interpolated = pd.DataFrame({'nproc': df1.nproc, 'SYPD': df1[method]})
    comp2_interpolated = pd.DataFrame({'nproc': df2.nproc, 'SYPD': df2[method]})

    c1_n = Component('IFS_n', comp1_interpolated.nproc, comp1_interpolated.SYPD, TTS_r, ETS_r)
    c2_n = Component('NEMO_n', comp2_interpolated.nproc, comp2_interpolated.SYPD, TTS_r, ETS_r)

    # Run LP model
    # find_optimal(c1_n, c2_n)

    from brute_force import brute_force
    optimal_result = brute_force(c1_n, c2_n, max_nproc)
    #
    # from iLP import solve_ilp
    # solve_ilp(c1_n, c2_n)

    print_result(c1_n, c2_n, optimal_result)
