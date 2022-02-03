import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot3d_fitness(c1_n, c2_n, fitness_mx, final_fitness, max_nproc):
    X, Y = np.meshgrid(c1_n.nproc, c2_n.nproc)
    Z = fitness_mx.to_numpy()

    Z_mk_same_sypd = final_fitness.to_numpy().T
    mk = X + Y > max_nproc
    Z_mk_max_nproc = Z.T * mk
    Z_mk_max_nproc[Z_mk_max_nproc == 0] = np.nan
    fig, ax = plt.subplots(4, 2, figsize=(8, 20), subplot_kw=dict(projection="3d"))
    for idx, angle in enumerate(range(181, 362, 30)):
        i = int(idx / 2)
        j = int(idx % 2)
        ax[i, j].plot_surface(X, Y, Z.T, cmap='viridis')
        ax[i, j].plot_surface(X, Y, Z_mk_same_sypd, color='gold')
        ax[i, j].plot_surface(X, Y, Z_mk_max_nproc, color='lightcoral', alpha=.5)
        ax[i, j].set_title("angle=" + str(angle))
        ax[i, j].set_xlabel(c1_n.name)
        ax[i, j].set_ylabel(c2_n.name)
        ax[i, j].set_zlabel('obj_f')
        ax[i, j].view_init(elev=20., azim=angle)

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
def get_cpl_cost(c1, c2, np1, np2):
    ts_c1 = c1.ts_info
    ts_c2 = c2.ts_info
    sp1 = c1.get_speedup(np1)
    sp2 = c2.get_speedup(np2)

    real_ts_c1 = ts_c1.Component * sp1 + ts_c1.Interpolation + ts_c1.Sending
    real_ts_c2 = ts_c2.Component * sp2 + ts_c2.Interpolation + ts_c2.Sending
    diff_real = real_ts_c1 - real_ts_c2

    # If we have irregular ts, we have to check which component is waiting
    c1_waiting = diff_real[diff_real < 0]  # c2 is faster --> c1 waits
    c2_waiting = diff_real[diff_real > 0]  # c1 is faster --> c2 waits
    waiting_cost_c1 = c1_waiting.sum() * np1
    waiting_cost_c2 = c2_waiting.sum() * np2

    return waiting_cost_c1 + waiting_cost_c2


def plot3d_cpl_cost(c1_n, c2_n, cpl_cost):
    X, Y = np.meshgrid(c1_n.nproc, c2_n.nproc)
    Z = cpl_cost.to_numpy()

    fig, ax = plt.subplots(4, 2, figsize=(8, 20), subplot_kw=dict(projection="3d"))
    for idx, angle in enumerate(range(181, 362, 30)):
        i = int(idx / 2)
        j = int(idx % 2)
        ax[i, j].plot_surface(X, Y, Z.T, cmap='viridis')
        ax[i, j].set_title("angle=" + str(angle))
        ax[i, j].set_xlabel(c1_n.name)
        ax[i, j].set_ylabel(c2_n.name)
        ax[i, j].set_zlabel('cpl_cost')
        ax[i, j].view_init(elev=20., azim=angle)

    ax[3, 1].plot_surface(X, Y, Z.T, cmap='viridis')
    ax[3, 1].set_title("Coupling cost")
    ax[3, 1].set_xlabel(c1_n.name)
    ax[3, 1].set_ylabel(c2_n.name)
    ax[3, 1].set_zlabel('cpl_cost')
    ax[3, 1].view_init(elev=270., azim=0.)
    plt.show()


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
        mask_max_nproc = c1_n.nproc <= max_nproc
        if c1_n.nproc_restriction.shape[0] > 0:
            mask_nproc_restriction = c1_n.nproc.isin(c1_n.nproc_restriction)
        else: mask_nproc_restriction = True
        rolling_mean = c1_n.fitness.fitness.rolling(10, center=True).mean() * mask_max_nproc * mask_nproc_restriction
        max_idx = rolling_mean.idxmax()
        opt_nproc = c1_n.fitness.nproc.iloc[max_idx]

        optimal_result = {
            "nproc_" + c1_n.name: opt_nproc,
            "fitness_" + c1_n.name: c1_n.get_fitness([opt_nproc]),
            "objective_f": c1_n.fitness.fitness.loc[max_idx],
            "SYPD": c1_n.get_sypd(opt_nproc),
        }

    elif num_components == 2:
        c1_n = list_components_class_interpolated[0]
        c2_n = list_components_class_interpolated[1]

        # TODO: check if ts_info is provided
        print("Using ts info")
        t0 = time.time()
        df_tmp = pd.DataFrame(index=c1_n.nproc, columns=c2_n.nproc)
        df_cpl_cost = df_tmp.apply(lambda x: get_cpl_cost(c1_n, c2_n, x.index, x.name))
        sim_ts_start = max(c1_n.ts_info.ts_id.min(), c2_n.ts_info.ts_id.min())
        sim_ts_end = min(c1_n.ts_info.ts_id.max(), c2_n.ts_info.ts_id.max())
        sim_time = sim_ts_end - sim_ts_start
        SY = sim_time / (365*24*3600)
        df_cpl_cost_ts = df_cpl_cost / SY
        print("execution time using ts info: ", time.time() - t0)

        if show_plots:
            plot3d_cpl_cost(c1_n, c2_n, df_cpl_cost)

        # SYPD difference matrix
        diff_tmp = pd.DataFrame(index=c1_n.sypd.SYPD, columns=c2_n.sypd.SYPD)
        diff_mx = diff_tmp.apply(lambda col: abs(col.name - col.index))
        diff_mx.columns = c2_n.nproc
        diff_mx.index = c1_n.nproc

        # Coupled fitness matrix
        f1 = c1_n.get_fitness(c1_n.nproc).fitness
        f2 = c2_n.get_fitness(c2_n.nproc).fitness
        fitness_mx_tmp = pd.DataFrame(index=f1, columns=f2)
        fitness_mx = fitness_mx_tmp.apply(lambda col: col.name + col.index)
        fitness_mx.index = c1_n.nproc
        fitness_mx.columns = c2_n.nproc


        # nproc matrix
        nproc_tmp = pd.DataFrame(index=c1_n.nproc, columns=c2_n.nproc)
        nproc_mx = nproc_tmp.apply(lambda col: col.name + col.index)

        # Filter to match the max_nproc restriction
        mask_max_nproc = nproc_mx <= max_nproc
        # Filter nproc_restriction per component
        if c1_n.nproc_restriction.shape[0] > 0:
            mask_nproc_restriction_c1 = c1_n.nproc.isin(c1_n.nproc_restriction)
        else:
            mask_nproc_restriction_c1 = True
        if c2_n.nproc_restriction.shape[0] > 0:
            mask_nproc_restriction_c2 = c2_n.nproc.isin(c2_n.nproc_restriction)
        else:
            mask_nproc_restriction_c2 = True


        # Filter only the combinations of processes of each component so that the difference of the SYPD is less than a threshold
        # TODO: Think the threshold parameter
        mask_same_SYPD = diff_mx < .3
        fitness_same_SYPD = fitness_mx[mask_same_SYPD]
        final_fitness = fitness_same_SYPD[mask_max_nproc]


        # TODO: Check this relative method
        filer_for_each_col = pd.Series(c1_n.sypd.SYPD.values * .01, index=c1_n.nproc)
        filter_col = diff_mx.le(filer_for_each_col, axis='index')
        filer_for_each_row = pd.Series(c2_n.sypd.SYPD.values * .01, index=c2_n.nproc)
        filter_row = diff_mx.le(filer_for_each_row, axis='columns')
        final_mask = filter_col * filter_row
        fitness_same_SYPD = fitness_mx[final_mask]
        final_fitness = fitness_same_SYPD[mask_max_nproc]


        # Filter only the combinations of processes of each component so that NEMO/IFS speed ratio is between 1.1 and 1.2
        # TODO: Think the IFS/NEMO balance
        # SYPD difference matrix
        diff_tmp = pd.DataFrame(index=c1_n.sypd.SYPD, columns=c2_n.sypd.SYPD)
        diff_mx = diff_tmp.apply(lambda col: col.name / col.index)
        diff_mx.columns = c2_n.nproc
        diff_mx.index = c1_n.nproc
        mask_same_SYPD = (diff_mx < 1.25) * (diff_mx > 1.10)
        fitness_same_SYPD = fitness_mx[mask_same_SYPD]
        final_fitness = fitness_same_SYPD[mask_max_nproc]

        # 3D Plot
        if show_plots:
            plot3d_fitness(c1_n, c2_n, fitness_mx, final_fitness, max_nproc)

        # Build the final solution
        c1_sum = final_fitness.sum(axis=1)
        c2_sum = final_fitness.sum(axis=0)
        count1 = final_fitness.count(axis=1)
        count2 = final_fitness.count(axis=0)
        df = pd.DataFrame(index=c1_n.nproc, columns=c2_n.nproc)
        rt = df.apply(lambda col: (c1_sum/count1 + c2_sum[col.name])/count2[col.name])
        rt_final = rt[mask_same_SYPD]
        rt_final = rt_final.mul(mask_nproc_restriction_c1, axis=1)
        rt_final = rt_final.mul(mask_nproc_restriction_c2, axis=0)
        nproc_c1 = rt_final.max(axis=1).idxmax()
        nproc_c2 = rt_final.max(axis=0).idxmax()

        # Sanity check: In case the combination of component processes does not map into a viable solution using max of
        # means, just select the maximum of the matrix
        if pd.isna(final_fitness.loc[nproc_c1, nproc_c2]):
            print("Singularity Error. Using the maximum of the fitness matrix")
            nproc_c1 = final_fitness.max(axis=1).idxmax()
            nproc_c2 = final_fitness.max(axis=0).idxmax()

        optimal_result = {
            "nproc_" + c1_n.name: nproc_c1,
            "nproc_" + c2_n.name: nproc_c2,
            "fitness_" + c1_n.name: c1_n.get_fitness([nproc_c1]),
            "fitness_" + c2_n.name: c2_n.get_fitness([nproc_c2]),
            "objective_f": final_fitness.loc[nproc_c1, nproc_c2],
            "SYPD": min(c1_n.get_sypd(nproc_c1), c2_n.get_sypd(nproc_c2)),
        }

    elif num_components == 3:
        c1_n = list_components_class_interpolated[0]
        c2_n = list_components_class_interpolated[1]
        c3_n = list_components_class_interpolated[2]
    return optimal_result

