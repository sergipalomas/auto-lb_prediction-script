import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


def minmax_df_normalization(df):
    columns_max = df.max()
    df_max = columns_max.max()
    columns_min = df.min()
    df_min = columns_min.min()
    normalized_df = (df - df_min) / (df_max - df_min)
    return normalized_df


def plot3d_fitness(c1_n, c2_n, fitness_mx, final_fitness, max_nproc):
    X, Y = np.meshgrid(c1_n.nproc, c2_n.nproc)
    Z = fitness_mx.to_numpy()

    Z_mk_same_sypd = final_fitness.to_numpy().T
    mk = X + Y > max_nproc
    Z_mk_max_nproc = Z.T * mk
    Z_mk_max_nproc[Z_mk_max_nproc == 0] = np.nan
    fig, ax = plt.subplots(4, 2, figsize=(8, 20), subplot_kw=dict(projection="3d"))
    for idx, rotation in enumerate(range(181, 362, 30)):
        i = int(idx / 2)
        j = int(idx % 2)
        ax[i, j].plot_surface(X, Y, Z.T, cmap='viridis')
        ax[i, j].plot_surface(X, Y, Z_mk_same_sypd, color='gold')
        ax[i, j].plot_surface(X, Y, Z_mk_max_nproc, color='lightcoral', alpha=.5)
        ax[i, j].set_title("rotation=" + str(rotation))
        ax[i, j].set_xlabel(c1_n.name)
        ax[i, j].set_ylabel(c2_n.name)
        ax[i, j].set_zlabel('obj_f')
        ax[i, j].view_init(elev=20., azim=rotation)

    ax[3, 1].plot_surface(X, Y, Z.T, cmap='viridis')
    ax[3, 1].plot_surface(X, Y, Z_mk_same_sypd, color='gold')
    ax[3, 1].plot_surface(X, Y, Z_mk_max_nproc, color='lightcoral')
    ax[3, 1].set_title("Objective Function")
    ax[3, 1].set_xlabel(c1_n.name)
    ax[3, 1].set_ylabel(c2_n.name)
    ax[3, 1].set_zlabel('obj_f')
    ax[3, 1].view_init(elev=270., azim=0.)
    plt.show()


# TODO: Check if irr ts cancel waiting time
# Returns the CoreHours wasted in coupling
def get_cpl_cost(c1, c2, np1, np2):
    ts_c1 = c1.ts_info
    ts_c2 = c2.ts_info
    # Get speedup for C1
    sp1 = c1.get_speedup(np1)
    sp1_ts_nproc = c1.get_speedup([c1.ts_nproc]).values
    sp1 = sp1 / sp1_ts_nproc
    # Get speedup for C2
    sp2 = c2.get_speedup([np2]).values
    sp2_ts_nproc = c2.get_speedup([c2.ts_nproc]).values
    sp2 = sp2 / sp2_ts_nproc

    # Use the speedup from the scalability curve to guess the ts_length for different nprocs than the provided in ts_info
    # Note that only ts.Component is rescaled. ts.Interpolation and ts.Sending are constant.
    df_tmp = pd.DataFrame(index=ts_c1.Component, columns=sp1)
    df_real_ts_c1 = df_tmp.apply(lambda x: x.index / x.name)
    df_real_ts_c1.index = ts_c1.ts_id
    real_ts_c2 = ts_c2.Component / sp2
    real_ts_c2.index = ts_c2.ts_id

    # Compute the diff (waiting time) between all pair of nprocs per timestep
    diff_real = df_real_ts_c1.sub(real_ts_c2, axis=0)

    # If we have irregular ts, we have to check which component is waiting
    c1_waiting = abs(diff_real[diff_real < 0])  # c2 is faster --> c1 waits
    c2_waiting = diff_real[diff_real > 0]  # c1 is faster --> c2 waits
    c1_cpl_time = c1_waiting.fillna(0).add((ts_c1.Interpolation + ts_c1.Sending).values, axis=0)
    c2_cpl_time = c2_waiting.fillna(0).add((ts_c2.Interpolation + ts_c2.Sending).values, axis=0)
    c1_cpl_ch = c1_cpl_time.sum() / 3600 * np1
    c2_cpl_ch = c2_cpl_time.sum() / 3600 * np2
    total_cpl_ch = c1_cpl_ch + c2_cpl_ch
    total_cpl_ch.index = np1

    return total_cpl_ch


def get_cpl_sypd(c1, c2, np1, np2):
    ts_c1 = c1.ts_info
    ts_c2 = c2.ts_info
    # sp1 = c1.get_speedup(np1)
    # sp2 = c2.get_speedup([np2]).values
    sp1 = c1.get_sypd_v(np1) / c1.get_sypd(c1.ts_nproc)
    sp2 = c2.get_sypd_v([np2]) / c2.get_sypd(c2.ts_nproc)

    # Use the speedup from the scalability curve to guess the ts_length for different nprocs than the provided in ts_info
    # Note that only ts.Component is rescaled. ts.Interpolation and ts.Sending are constant.
    df_tmp = pd.DataFrame(index=ts_c1.Component, columns=sp1)
    df_real_ts_c1 = df_tmp.apply(lambda x: x.index / x.name + ts_c1.Interpolation + ts_c1.Sending)
    real_ts_c2 = ts_c2.Component / sp2.values + ts_c2.Interpolation + ts_c2.Sending

    # Compute the diff (waiting time) between all pair of nprocs per timestep
    diff_real = df_real_ts_c1.sub(real_ts_c2, axis=0).shift(1)
    diff_real.fillna(diff_real.mean())

    # If we have irregular ts, we have to check which component is waiting
    c1_waiting = abs(diff_real[diff_real < 0])  # c2 is faster --> c1 waits
    c2_waiting = diff_real[diff_real > 0]  # c1 is faster --> c2 waits
    c1_sim_time = df_real_ts_c1.add(c1_waiting, fill_value=0).sum()
    c2_sim_time = c2_waiting.fillna(0).add(real_ts_c2, axis=0).sum()
    if not c1_sim_time.equals(c2_sim_time):
        warnings.warn(
            "Component 1 and 2 got different results when computing the coupled execution time! (using ts info")
    c1_sim_time.index = np1
    return c1_sim_time


def plot3d_cpl_cost(c1_n, c2_n, cpl_cost):
    X, Y = np.meshgrid(c1_n.nproc, c2_n.nproc)
    Z = cpl_cost.to_numpy()

    fig, ax = plt.subplots(4, 2, figsize=(8, 20), subplot_kw=dict(projection="3d"))
    for idx, rotation in enumerate(range(181, 362, 30)):
        i = int(idx / 2)
        j = int(idx % 2)
        ax[i, j].plot_surface(X, Y, Z.T, cmap='viridis')
        ax[i, j].set_title("rotation=" + str(rotation))
        ax[i, j].set_xlabel(c1_n.name)
        ax[i, j].set_ylabel(c2_n.name)
        ax[i, j].set_zlabel('cpl_cost')
        ax[i, j].view_init(elev=20., azim=rotation)

    ax[3, 1].plot_surface(X, Y, Z.T, cmap='viridis')
    ax[3, 1].set_title("Coupling cost")
    ax[3, 1].set_xlabel(c1_n.name)
    ax[3, 1].set_ylabel(c2_n.name)
    ax[3, 1].set_zlabel('cpl_cost')
    ax[3, 1].view_init(elev=270., azim=0.)
    plt.show()


def plot_timesteps(component):
    ts = component.ts_info
    ts.Component.plot(style='b.')
    mean = [ts.Component.mean()] * ts.Component.shape[0]
    plt.plot(mean, label='mean')
    plt.title(component.name + " timestep length distribution")
    plt.show()


# Method that takes into account the speedup achieved and the efficiency
# Proportional to SY/CH metric
def get_performance_efficiency_metric(df_TTS):
    base_case_sypd = df_TTS.iloc[0, 0]
    df_cpl_speedup = df_TTS.divide(base_case_sypd)
    base_case_nproc = df_TTS.index[0] + df_TTS.columns[0]
    df_cpl_efficiency = df_cpl_speedup.apply(lambda x: x / ((x.index + x.name) / base_case_nproc))
    return df_cpl_speedup * df_cpl_efficiency


def brute_force(num_components, list_components_class_interpolated, max_nproc, show_plots):
    # Sanity check for max_nproc parameter
    sum_max = 0
    sum_min = 0
    for component in list_components_class_interpolated:
        sum_max += component.nproc.max()
        sum_min += component.nproc.min()
    if max_nproc == 0 or max_nproc > sum_max:
        max_nproc = sum_max
    if sum_min <= max_nproc <= max_nproc:
        print("Using a limitation of %i processes at most." % max_nproc)
    elif max_nproc <= sum_min:
        print("Error: max_nproc is less than the minimum resource configuration provided in the CSV files")
        exit(1)

    if num_components == 1:
        c1_n = list_components_class_interpolated[0]
        # mask_max_nproc = c1_n.nproc <= max_nproc
        # if c1_n.nproc_restriction.shape[0] > 0:
        #     mask_nproc_restriction = c1_n.nproc.isin(c1_n.nproc_restriction)
        # else:
        #     mask_nproc_restriction = True

        # rolling_mean = c1_n.fitness.fitness.rolling(max(2, round(c1_n.nproc.shape[0]*0.15)), center=True).mean() * mask_max_nproc * mask_nproc_restriction
        # max_idx = rolling_mean.idxmax()
        opt_nproc = c1_n.fitness.nproc[c1_n.fitness.fitness.idxmax()]

        optimal_result = {
            "nproc_" + c1_n.name: opt_nproc,
            "fitness_" + c1_n.name: c1_n.get_fitness([opt_nproc]),
            "objective_f": c1_n.fitness.fitness[c1_n.fitness.nproc == opt_nproc],
            "SYPD": c1_n.get_sypd(opt_nproc),
        }

    # n-component case
    else:
        nproc_list = []
        sypd_list = []
        chsy_list = []

        for component in list_components_class_interpolated:
            nproc_list.append(component.nproc.values)
            sypd_list.append(component.sypd.SYPD)
            chsy_list.append(component.chsy.CHSY)

        # Get nproc combinations and total
        comb_nproc = np.array(np.meshgrid(*nproc_list)).T.reshape(-1, num_components)
        cpl_nproc = comb_nproc.sum(axis=1)

        # Get SYPD combinations and minimum
        comb_sypd = np.array(np.meshgrid(*sypd_list)).T.reshape(-1, num_components)
        cpl_sypd = comb_sypd.min(axis=1)

        # Get the CHSY if components were executed in standalone
        comb_chsy = np.array(np.meshgrid(*chsy_list)).T.reshape(-1, num_components)
        components_chsy = comb_chsy.sum(axis=1)

        # Get CHSY expected.  Use equation CHSY = 24*NP/SYPD
        cpl_chsy = 24 * cpl_nproc / cpl_sypd

        # Get the Coupling cost dividing the component's standalone CHSY by the expected coupling CHSY
        cpl_cost = 1 - components_chsy / cpl_chsy
        cpl_cost_chsy = cpl_cost * cpl_chsy

        # Filter using EDP > basecase & max_nproc
        # perf_eff_metric = get_performance_efficiency_metric(df_TTS)
        base_case_sypd = cpl_sypd[0]
        base_case_nproc = cpl_nproc[0]
        cpl_speedup = cpl_sypd / base_case_sypd
        cpl_efficiency = cpl_speedup / (cpl_nproc / base_case_nproc)
        perf_eff_metric = cpl_speedup * cpl_efficiency
        mask_better_basecase = perf_eff_metric >= 1
        # Maks to match the max_nproc restriction
        mask_max_nproc = cpl_nproc <= max_nproc
        # Compute the fitness
        mask = mask_max_nproc & mask_better_basecase
        cpl_sypd_mk = cpl_sypd[mask]
        cpl_chsy_mk = cpl_chsy[mask]
        cpl_nproc_mk = cpl_nproc[mask]
        comb_nproc_mk = comb_nproc[mask]
        cpl_cost_mk = cpl_cost[mask]
        cpl_cost_chsy_mk = cpl_cost_chsy[mask]

        f_TTS = minmax_df_normalization(cpl_sypd_mk)
        f_ETS = 1 - minmax_df_normalization(cpl_chsy_mk)
        final_fitness = component.TTS_r * f_TTS + component.ETS_r * f_ETS

        # Pick up the best resource configuration and the top5
        top_idx = final_fitness.argmax()
        top5_idx = np.argpartition(final_fitness, -5)[-5:]

        # Build the top solution
        top_cpl_fitness = final_fitness[top_idx]
        top_cpl_nproc = comb_nproc_mk[top_idx]
        top_cpl_sypd = cpl_sypd_mk[top_idx]
        top_cpl_cplc = cpl_cost_mk[top_idx]
        top_cpl_chsy = cpl_cost_chsy_mk[top_idx]

        # Build the top5 solution
        top5_cpl_fitness = final_fitness[top5_idx]
        top5_cpl_nproc = comb_nproc_mk[top5_idx]

        for i, component in enumerate(list_components_class_interpolated):
            component.top_nproc = top_cpl_nproc[i]
            component.top5_nproc = top5_cpl_nproc[:, i]

        coupled_optimal_result = {
            "objective_f": final_fitness[top_idx],
            "SYPD": top_cpl_sypd,
            "cpl_cost": top_cpl_cplc,
            "cpl_chsy": top_cpl_chsy,
        }

    return coupled_optimal_result
