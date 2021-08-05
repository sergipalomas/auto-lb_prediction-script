import pandas as pd
from component_class import sypd2chpsy
import matplotlib.pyplot as plt
from component_class import Component

def calc_fitness(sypd, chpsy, TTS_r, ETS_r):
    return TTS_r * sypd + ETS_r * chpsy

def brute_force(c1, c2):

    # Sort before matching
    c1_sort = c1.sypd.sort_values('SYPD')
    c2_sort = c2.sypd.sort_values('SYPD')

    # For each combination nproc/SYPD of c1, give the nproc of c2 which minimizes the differences in the SYPD between
    # both components
    closest = pd.merge_asof(c1_sort, c2_sort, on='SYPD', direction='nearest')
    f1 = closest.apply(lambda x: c1.get_fitness(x['nproc_x']), axis=1)
    f2 = closest.apply(lambda x: c2.get_fitness(x['nproc_y']), axis=1)
    closest['objective_f'] = f1 + f2
    optimal_result = closest.iloc[closest.objective_f.idxmax()]
    plt.plot(closest.SYPD, closest.objective_f)
    plt.plot(optimal_result.SYPD, optimal_result.objective_f, 'o')
    plt.title("Objective function")
    plt.show()
    optimal_result = closest.iloc[closest.objective_f.idxmax()]
    print("Optimal: \n")
    print(optimal_result)
    plt.plot(c1.nproc, c1.fitness.fitness)
    plt.plot(c2.nproc, c2.fitness.fitness)
    plt.plot(optimal_result.nproc_x, c1.get_fitness(optimal_result.nproc_x), 'o')
    plt.plot(optimal_result.nproc_y, c2.get_fitness(optimal_result.nproc_y), 'o')
    plt.title("Fitness value")
    plt.legend(['IFS', 'NEMO'])
    plt.show()

    c1.plot_scalability()
    c2.plot_scalability()