import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_fitness(sypd, chpsy, TTS_r, ETS_r):
    return TTS_r * sypd + ETS_r * chpsy


def obj_f2(nproc, c1_n, c2_n):
    nproc1 = nproc[0]
    nproc2 = nproc[1]
    r = np.vectorize(c1_n.get_fitness)(nproc1) + np.vectorize(c2_n.get_fitness)(nproc2)
    return r


def plot_obj(c1_n, c2_n, max_nproc, opt_result):
    nproc1_n = np.linspace(c1_n.nproc.min(), c1_n.nproc.max(), 75).round()
    nproc2_n = np.linspace(c2_n.nproc.min(), c2_n.nproc.max(), 75).round()

    X, Y = np.meshgrid(nproc1_n, nproc2_n)
    Z = obj_f2([X, Y], c1_n, c2_n)

    # Create a mask for Z when the number of processes exceeds max_numproc
    mk = X + Y > max_nproc
    Z_mk = Z * mk
    Z_mk[Z_mk == 0] = np.nan

    # Prepare optimal point
    x_opt = [opt_result['nproc_' + c1_n.name], opt_result['nproc_' + c1_n.name]]
    y_opt = [opt_result['nproc_' + c2_n.name], opt_result['nproc_' + c2_n.name]]
    z_opt = [Z.min(), opt_result['objective_f']]

    fig, ax = plt.subplots(4, 2, figsize=(8, 20), subplot_kw=dict(projection="3d"))
    for idx, angle in enumerate(range(181, 362, 30)):
        i = int(idx / 2)
        j = int(idx % 2)
        ax[i, j].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=.9)
        ax[i, j].plot_surface(X, Y, Z_mk, color='lightcoral', edgecolor='none')
        ax[i, j].set_title("angle=" + str(angle))
        ax[i, j].set_xlabel(c1_n.name)
        ax[i, j].set_ylabel(c2_n.name)
        ax[i, j].set_zlabel('obj_f')
        ax[i, j].plot(x_opt, y_opt, z_opt, 'r--')
        ax[i, j].plot(x_opt[1], y_opt[1], z_opt[1], 'ro')
        ax[i, j].view_init(elev=20., azim=angle)

    ax[3, 1].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=.9)
    ax[3, 1].plot_surface(X, Y, Z_mk, color='lightcoral', edgecolor='none')
    ax[3, 1].plot(x_opt[1], y_opt[1], z_opt[1], 'ro')
    ax[3, 1].set_title("vertical")
    ax[3, 1].set_xlabel(c1_n.name)
    ax[3, 1].set_ylabel(c2_n.name)
    ax[3, 1].set_zlabel('obj_f')

    ax[3, 1].view_init(elev=270., azim=0.)
    plt.show()


def brute_force_old(c1_n, c2_n, max_nproc):
    # Sanity check for max_nproc parameter
    if max_nproc == 0 or max_nproc > c1_n.nproc.max() + c2_n.nproc.max():
        max_nproc = c1_n.nproc.max() + c2_n.nproc.max()
        print("Using a limitation of %i processes at most." % max_nproc)
    if max_nproc < c1_n.nproc.min() + c2_n.nproc.min():
        print("Error: max_nproc is less than the minimum resource configuration provided in the CSV files")
        exit(1)
    else:
        print("Using a limitation of %i processes at most." % max_nproc)

    # Sort before matching
    c1_n_sort = c1_n.sypd.sort_values('SYPD')
    c2_n_sort = c2_n.sypd.sort_values('SYPD')

    closest = pd.merge_asof(c1_n_sort, c2_n_sort, on='SYPD', direction='nearest')
    closest.rename(columns={'nproc_x': 'nproc_' + c1_n.name, 'nproc_y': 'nproc_' + c2_n.name}, inplace=True)
    closest['f1'] = closest.apply(lambda x: c1_n.get_fitness(x['nproc_' + c1_n.name]), axis=1)
    closest['f2'] = closest.apply(lambda x: c2_n.get_fitness(x['nproc_' + c2_n.name]), axis=1)
    closest['objective_f'] = closest.f1 + closest.f2
    if max_nproc > 0:
        closest = closest[closest['nproc_' + c1_n.name] + closest['nproc_' + c2_n.name] < max_nproc]
    optimal_result = closest.iloc[closest.objective_f.idxmax()]
    plt.plot(closest.SYPD, closest.objective_f)
    plt.plot(optimal_result.SYPD, optimal_result.objective_f, 'o')
    plt.title("Objective function")
    plt.show()

    opt = closest.iloc[closest.objective_f.idxmax()]

    optimal_result = {
        "nproc_" + c1_n.name: opt['nproc_' + c1_n.name],
        "nproc_" + c2_n.name: opt['nproc_' + c2_n.name],
        "fitness_" + c1_n.name: opt.f1,
        "fitness_" + c2_n.name: opt.f2,
        "objective_f": opt.objective_f,
        "SYPD": opt.SYPD,
    }

    plot_obj(c1_n, c2_n, max_nproc, optimal_result)

    return optimal_result


def brute_force(list_components_class_interpolated, max_nproc):

    # Sanity check for max_nproc parameter
    sum_max = 0
    sum_min = 0
    for component in list_components_class_interpolated:
        sum_max += component.nproc.max()
        sum_min += component.nproc.min()
    if max_nproc == 0 or max_nproc > sum_max:
        max_nproc = sum_max
        print("Using a limitation of %i processes at most." % max_nproc)
    if max_nproc < sum_min:
        print("Error: max_nproc is less than the minimum resource configuration provided in the CSV files")
        exit(1)
    else:
        print("Using a limitation of %i processes at most." % max_nproc)

    c1_n = list_components_class_interpolated[0]
    c2_n = list_components_class_interpolated[1]
    sypd_c1 = c1_n.nproc.apply(lambda x: c1_n.get_sypd(x))
    sypd_c2 = c2_n.nproc.apply(lambda x: c2_n.get_sypd(x))
    diff_tmp = pd.DataFrame(index=sypd_c1, columns=sypd_c2)
    diff_mx = diff_tmp.apply(lambda col: abs(col.name - col.index))
    f1 = c1_n.get_fitness2(c1_n.nproc).fitness
    f2 = c2_n.get_fitness2(c2_n.nproc).fitness
    fitness_mx_tmp = pd.DataFrame(index=f1, columns=f2)
    fitness_mx = fitness_mx_tmp.apply(lambda col: col.index + col.name)
    diff_mx.columns = c2_n.nproc
    diff_mx.index = c1_n.nproc
    fitness_mx.columns = c2_n.nproc
    fitness_mx.index = c1_n.nproc
    # Filter only the combinations of processes of each component so that the difference of the SYPD is less than a threshold
    # TODO: Think the threshold parameter
    mask = diff_mx < 3
    final_abs = diff_mx[mask]
    final_fitness = fitness_mx[mask]

    # TODO: Check this relative method
    # diff_mx2 = diff_tmp.apply(lambda col: abs(col.name - col.index) / col.name)
    # final_rel = diff_mx2[diff_mx2 < 0.03]

    # 3D Plot
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


    # Build the final solution
    nproc_c1 = final_fitness.mean(axis=1).idxmax()
    nproc_c2 = final_fitness.mean(axis=0).idxmax()

    # Sanity check: In case the combination of component processes does not map into a viable solution using max of
    # means, just select the maximum of the matrix
    if final_fitness.loc[nproc_c1, nproc_c2] == np.nan:
        print("Singularity Error. Using the maximum of the fitness matrix")
        nproc_c1 = final_fitness.max(axis=1).idxmax()
        nproc_c2 = final_fitness.max(axis=0).idxmax()

    optimal_result = {
        "nproc_" + c1_n.name: nproc_c1,
        "nproc_" + c2_n.name: nproc_c2,
        "fitness_" + c1_n.name: c1_n.get_fitness(nproc_c1),
        "fitness_" + c2_n.name: c2_n.get_fitness(nproc_c2),
        "objective_f": final_fitness.loc[nproc_c1, nproc_c2],
        "SYPD": min(c1_n.get_sypd(nproc_c1), c2_n.get_sypd(nproc_c2)),
    }

    return optimal_result
