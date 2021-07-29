import pandas as pd
from component_class import sypd2chpsy
import matplotlib.pyplot as plt
from component_class import Component

def calc_fitness(sypd, chpsy, TTS_r, ETS_r):
    return TTS_r * sypd + ETS_r * chpsy

def brute_force(c1, c2, step):

    # Sort before matching
    c1_sort = c1.sypd.sort_values('SYPD')
    c2_sort = c2.sypd.sort_values('SYPD')

    # For each combination nproc/SYPD of c1, give the nproc of c2 which minimizes the differences in the SYPD between
    # both components
    closest = pd.merge_asof(c1_sort, c2_sort, on='SYPD', direction='nearest')
    f1 = c1.get_fitness2(closest.nproc_x).fitness
    f2 = c2.get_fitness2(closest.nproc_y).fitness
    closest['objective_f'] = f1 + f2
    plt.plot(closest.SYPD, closest.objective_f)
    plt.title("Objective function")
    plt.show()
    optimal_result = closest.iloc[closest.objective_f.idxmax()]
    print("Optimal: \n")
    print(optimal_result)
    plt.plot(c1.nproc, c1.fitness)
    plt.plot(c2.nproc, c2.fitness)
    plt.plot(optimal_result.nproc_x, c1.get_fitness(optimal_result.nproc_x), 'o')
    plt.plot(optimal_result.nproc_y, c2.get_fitness(optimal_result.nproc_y), 'o')
    plt.show()
