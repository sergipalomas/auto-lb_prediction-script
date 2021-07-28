import pandas as pd
from component_class import sypd2chpsy
import matplotlib.pyplot as plt

def calc_fitness(sypd, chpsy, TTS_r, ETS_r):
    return TTS_r * sypd + ETS_r * chpsy

def brute_force(c1, c2, c1_df, c2_df, step):
    # Find bottleneck component
    if c1.max_sypd >= c2.max_sypd:
        bc = c2
    else: bc = c1

    # Sort before matching
    c1_sort = c1_df.sort_values('SYPD')
    c2_sort = c2_df.sort_values('SYPD')

    # Find nearest SYPD for each component nproc configuration
    closest = pd.merge_asof(c1_sort, c2_sort, on='SYPD', direction='nearest')
    closest['chpsy_x'] = sypd2chpsy(closest.nproc_x, closest.SYPD)
    closest['chpsy_y'] = sypd2chpsy(closest.nproc_y, closest.SYPD)
    closest['fitness_x'] = calc_fitness(closest.SYPD, closest.chpsy_x, c1.TTS_r, c1.ETS_r)
    closest['fitness_y'] = calc_fitness(closest.SYPD, closest.chpsy_y, c2.TTS_r, c2.ETS_r)
    closest['objective_f'] = (closest.nproc_x * closest.fitness_x  + closest.nproc_y *
                              closest.fitness_y) / (closest.nproc_x + closest.nproc_y)
    plt.plot(closest.SYPD, closest.objective_f)
    plt.show()
    optimal_result = closest.iloc[closest.objective_f.idxmax()]
    print("Optimal: \n")
    print(optimal_result)
    plt.plot(c1.nproc, c1.fitness)
    plt.plot(c2.nproc, c2.fitness)
    plt.plot(optimal_result.nproc_x, c1.get_fitness(optimal_result.nproc_x), 'o')
    plt.plot(optimal_result.nproc_y, c2.get_fitness(optimal_result.nproc_y), 'o')
    plt.show()
