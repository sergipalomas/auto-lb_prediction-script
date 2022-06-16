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
    sp1 = c1.get_speedup(np1)
    sp2 = c2.get_speedup([np2]).values

    # Use the speedup from the scalability curve to guess the ts_length for different nprocs than the provided in ts_info
    # Note that only ts.Component is rescaled. ts.Interpolation and ts.Sending are constant.
    df_tmp = pd.DataFrame(index=ts_c1.Component, columns=sp1)
    df_real_ts_c1 = df_tmp.apply(lambda x: x.index / x.name + ts_c1.Interpolation + ts_c1.Sending)
    real_ts_c2 = ts_c2.Component / sp2 + ts_c2.Interpolation + ts_c2.Sending

    # Compute the diff (waiting time) between all pair of nprocs per timestep
    diff_real = df_real_ts_c1.sub(real_ts_c2, axis=0)

    # If we have irregular ts, we have to check which component is waiting
    c1_waiting = abs(diff_real[diff_real < 0])  # c2 is faster --> c1 waits
    c2_waiting = diff_real[diff_real > 0]       # c1 is faster --> c2 waits
    c1_cpl_time = c1_waiting.fillna(0).add(ts_c1.Interpolation + ts_c1.Sending, axis=0)
    c2_cpl_time = c2_waiting.fillna(0).add(ts_c2.Interpolation + ts_c2.Sending, axis=0)
    c1_cpl_cost = c1_cpl_time.sum()/3600 * np1
    c2_cpl_cost = c2_cpl_time.sum()/3600 * np2
    total_cpl_cost = c1_cpl_cost + c2_cpl_cost
    total_cpl_cost.index = np1

    return total_cpl_cost

def get_cpl_sypd(c1, c2, np1, np2):
    ts_c1 = c1.ts_info
    ts_c2 = c2.ts_info
    #sp1 = c1.get_speedup(np1)
    #sp2 = c2.get_speedup([np2]).values
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
    c2_waiting = diff_real[diff_real > 0]       # c1 is faster --> c2 waits
    c1_sim_time = df_real_ts_c1.add(c1_waiting, fill_value=0).sum()
    c2_sim_time = c2_waiting.fillna(0).add(real_ts_c2, axis=0).sum()
    if not c1_sim_time.equals(c2_sim_time):
        warnings.warn("Component 1 and 2 got different results when computing the coupled execution time! (using ts info")
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


def plot_timesteps_IFS(component):
    ts = component.ts_info
    ts_length = ts.drop('ts_id', axis=1).sum(axis=1)
    irr_ts_idx = ts_length.groupby(np.arange(len(ts_length)) // 4).idxmax()
    irr_ts = [x for x in ts.Component if x in irr_ts_idx]
    reg_ts = [x for x in ts.Component if x not in irr_ts_idx]
    ts.Component.plot(style='b.', label='Regular ts')
    ts.Component[3::4].plot(style='r.', label='Irregular ts')
    mean = [ts.Component.mean()] * ts.Component.shape[0]
    plt.plot(mean, label='mean')
    plt.title(component.name + " timestep length distribution")
    plt.legend()
    plt.show()


def plot_timesteps(component):
    ts = component.ts_info
    ts.Component.plot(style='b.')
    mean = [ts.Component.mean()] * ts.Component.shape[0]
    plt.plot(mean, label='mean')
    plt.title(component.name + " timestep length distribution")
    plt.show()


    # Method that takes into account the speedup achieved and the efficiency
    # Proportional to SY/CH throughput metric
def get_performance_efficiency_metric(df_TTS):
    base_case_sypd = df_TTS.iloc[0, 0]
    df_cpl_speedup = df_TTS.divide(base_case_sypd)
    base_case_nproc = df_TTS.index[0] + df_TTS.columns[0]
    df_cpl_efficiency = df_cpl_speedup.apply(lambda x: x / ((x.index + x.name) / base_case_nproc))
    return df_cpl_speedup * df_cpl_efficiency


def new_brute_force(num_components, list_components_class_interpolated, max_nproc, show_plots):
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

    elif num_components == 2:

        c1_n = list_components_class_interpolated[0]
        c2_n = list_components_class_interpolated[1]

        df_nproc_tmp = pd.DataFrame(index=c1_n.nproc, columns=c2_n.nproc)
        df_sypd_tmp = pd.DataFrame(index=c1_n.sypd.SYPD, columns=c2_n.sypd.SYPD)
        df_chsy_tmp = pd.DataFrame(index=c1_n.chsy.CHPSY, columns=c2_n.chsy.CHPSY)

        # Maks to match the max_nproc restriction
        df_nproc = df_nproc_tmp.apply(lambda col: col.name + col.index)
        mask_max_nproc = df_nproc <= max_nproc

        ### TTS matrix (in SYPD)
        df_TTS = df_sypd_tmp.apply(lambda col: np.minimum(col.index, col.name))
        df_TTS.index = c1_n.nproc
        df_TTS.columns = c2_n.nproc

        ### ETS matrix (in CHSY)
        df_ETS = df_TTS.apply(lambda x: (x.name + x.index)*24/x)

        # CHPSY matrix
        if len(c1_n.ts_info != 0) and len(c2_n.ts_info != 0):
            # Use information per timestep to find best combination of processes
            print("Using timestep lengths information")
            t0 = time.time()
            if show_plots:
                plot_timesteps_IFS(c1_n)
                plot_timesteps(c2_n)

            # Get some information from the ts_info
            sim_ts_start = min(c1_n.ts_info.ts_id.min(), c2_n.ts_info.ts_id.min())
            sim_ts_end = max(c1_n.ts_info.ts_id.max(), c2_n.ts_info.ts_id.max())
            sim_time = sim_ts_end - sim_ts_start
            SY = sim_time / (365 * 24 * 3600)

            # cpl_cost CHPSY
            df_cpl_cost_ch = df_nproc_tmp.apply(lambda x: get_cpl_cost(c1_n, c2_n, x.index, x.name))
            df_cpl_cost_chsy = df_cpl_cost_ch / SY

            # cpl CHSY
            #df_ETS = df_chsy.add(abs(df_cpl_cost_chsy)) !TODO: REDO
            cpl_cost = df_cpl_cost_chsy / df_ETS

            print("execution time using ts info: ", time.time() - t0)

        else:
            # Assume regular timestep lengths
            print("No information per timestep available. Assuming regular timestep lengths")
            ratio = df_sypd_tmp.apply(lambda col: col.index / col.name)
            c1_waits = 1 - 1 / ratio[ratio > 1]
            c2_waits = 1 - ratio[ratio < 1]
            coupled_c2_chsy = df_ETS.mul(c2_n.nproc.values / df_nproc, axis=0)
            coupled_c1_chsy = df_ETS - coupled_c2_chsy
            c1_waits_cost = c1_waits.values*coupled_c1_chsy
            c2_waits_cost = c2_waits.values*coupled_c2_chsy
            df_cpl_cost_chsy = c1_waits_cost.add(c2_waits_cost, fill_value=0)

            ### ETS matrix
            # Add cpl_cost chsy overhead to ETS matrix
            cpl_cost = df_cpl_cost_chsy / df_ETS


        if show_plots:
            plot3d_cpl_cost(c1_n, c2_n, df_cpl_cost_chsy)

        perf_eff_metric = get_performance_efficiency_metric(df_TTS)
        mask_best_results = perf_eff_metric >= perf_eff_metric.stack().quantile(.25)

        #df_ETS_stacked = df_ETS.stack()
        #df_ETS_tmp = df_ETS_stacked[np.abs(df_ETS_stacked-df_ETS_stacked.mean()) <= 2*df_ETS_stacked.std()].unstack()

        df_top_TTS = df_TTS[mask_best_results & mask_max_nproc]
        df_top_ETS = df_ETS[mask_best_results & mask_max_nproc]

        # Min/Max normalization
        f_TTS = minmax_df_normalization(df_top_TTS)
        # Note that we want to minimize the cost. Therefore, I use 1 - cost to have a maximization problem
        f_ETS = 1 - minmax_df_normalization(df_top_ETS)

        # Objective Function
        final_fitness = c1_n.TTS_r * f_TTS + c1_n.ETS_r * f_ETS


        # Filter by max_nproc allowed
        mask_better_basecase = perf_eff_metric >= 1
        df_good_TTS = df_TTS[mask_better_basecase & mask_max_nproc]
        df_good_ETS = df_ETS[mask_better_basecase & mask_max_nproc]
        f_TTS_new = minmax_df_normalization(df_good_TTS)
        f_ETS_new = 1 - minmax_df_normalization(df_good_ETS)
        new_final_fitness = c1_n.TTS_r * f_TTS_new + c1_n.ETS_r * f_ETS_new

        top10_fitness = final_fitness.stack().nlargest(10)
        top10_newfitness = new_final_fitness.stack().nlargest(10)
        top10_perf_eff = perf_eff_metric[mask_max_nproc].stack().nlargest(10)

        final_fitness = new_final_fitness[mask_max_nproc]

        # Create the final solution by averaging each result with its closest 4 neighbours (left, right, up, down)
        # I don't use this if there is a nproc_restriction. As the jump between consecutive nprocs can be too big and
        # averaging would be unfair.
        #if len(c1_n.nproc_restriction) == 0 and len(c2_n.nproc_restriction) == 0:
        if False:
            row_rolling_mean = final_fitness.rolling(3, center=True, min_periods=2, axis=1).mean()
            col_rolling_mean = final_fitness.rolling(3, center=True, min_periods=2, axis=0).mean()
            f = (row_rolling_mean + col_rolling_mean) / 2
            nproc_c1 = f.max(axis=1).idxmax()
            nproc_c2 = f.max(axis=0).idxmax()

        # Just pick up the maximum from the table.
        else:
            nproc_c1, nproc_c2 = final_fitness.stack().idxmax()
            top_configurations = final_fitness.stack().nlargest(5).index

        # TODO: Fix cpl cost and chsy output
        optimal_result = {
            "nproc_" + c1_n.name: nproc_c1,
            "nproc_" + c2_n.name: nproc_c2,
            "fitness_" + c1_n.name: c1_n.get_fitness([nproc_c1]),
            "fitness_" + c2_n.name: c2_n.get_fitness([nproc_c2]),
            "objective_f": final_fitness.loc[nproc_c1, nproc_c2],
            "SYPD": df_TTS.loc[nproc_c1, nproc_c2],
            "cpl_cost": cpl_cost.loc[nproc_c1, nproc_c2],
            "cpl_chsy": df_cpl_cost_chsy.loc[nproc_c1, nproc_c2],
            "speed_ratio": c1_n.get_sypd(nproc_c1)/c2_n.get_sypd(nproc_c2),
            "top_configurations": top_configurations
        }

    return optimal_result
