import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import sys
import yaml
import os
mpl.use("Agg")


def check_interpo(num_components, list_components_class, list_components_scalability_df):

    for i in range(num_components):

        c1 = list_components_class[i]

        # Plot the SYPD
        fig1, ax1 = plt.subplots()
        c1.sypd.plot(x='nproc', y='SYPD', ax=ax1)

        # Plot the fitness
        fig2, ax2 = plt.subplots()
        c1.fitness.plot(x='nproc', y='fitness', ax=ax2)

        methods = ['linear', 'slinear', 'quadratic'] #, 'cubic']
        legend = methods.copy()
        legend.insert(0, 'Real')
        df1 = list_components_scalability_df[i]
        for m in methods:
            ### Start using interpolated data
            comp1_new = pd.concat({'nproc': df1.nproc, 'SYPD': df1[m]})
            t1 = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD, c1.nproc_restriction, c1.ts_info, c1.ts_nproc, TTS_r, ETS_r)

            t1.sypd.plot(x='nproc', y='SYPD', ax=ax1)
            t1.fitness.plot(x='nproc', y='fitness', ax=ax2)

        ax1.set_title(c1.name + " SYPD after interpolating")
        ax1.set_xlabel('nproc')
        ax1.set_ylabel('SYPD')
        ax1.legend(legend)

        ax2.set_title(c1.name + " Fitness after interpolating")
        ax2.set_xlabel('nproc')
        ax2.set_ylabel('fitness')
        ax2.legend(legend)
        fig_name = c1.name + "_SYPD_Fitness_after_interpolating.png"
        plt.savefig("./img/" + fig_name)
        #plt.show()



def interpolate_data(component, nproc_step):
    nproc_start = component.nproc.min()
    nproc_end = component.nproc.max() + 1
    ## Interpolation
    methods = ['linear', 'slinear', 'quadratic']#, 'cubic']
    legend = methods.copy()
    legend.insert(0, 'real')

    ## Interpolation
    if len(component.nproc_restriction) != 0:
        xnew = component.nproc_restriction.values
    else:
        xnew = np.arange(nproc_start, nproc_end, nproc_step)
    if len(component.ts_info) != 0 and component.ts_nproc not in xnew:
        xnew = np.append(xnew, component.ts_nproc)
        xnew.sort()
    tmp_component = pd.Series([component.sypd.SYPD[component.nproc[component.nproc == n].index[0]]
                               if n in component.nproc.values else np.NaN for n in xnew])
    df = pd.DataFrame({'nproc': xnew, 'real': tmp_component})
    for m in methods:
        f = interpolate.interp1d(component.nproc, component.sypd.SYPD, kind=m, fill_value="extrapolate")
        ynew = f(xnew).round(2)
        df[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(component.nproc, component.sypd.SYPD, 'o')
        for m in methods:
            plt.plot(xnew, df[m])
        plt.legend(legend)
        plt.title("Check interpo " + component.name)
        fig_name = component.name + "_check_interpo.png"
        plt.savefig(fig_name)
        #plt.show()

    return df


def print_result(num_components, list_components_class_interpolated, optimal_result):

    print("Optimal for TTS=%.1f, ETS=%.1f:" % (TTS_r, ETS_r))
    nproc_acc = 0
    chpsy_acc = 0
    for component in list_components_class_interpolated:
        opt_nproc = optimal_result['nproc_' + component.name]
        print("\n -------------------------------\n")
        print("Results for component %s:" % component.name)
        print("Number of processes: %i" % opt_nproc)
        print("Fitness: %.2f" % component.get_fitness([opt_nproc]).fitness)
        print("CHPSY: %i" % (component.get_chpsy(opt_nproc)))
        print("SYPD: %.2f" % component.get_sypd(opt_nproc))
        component.plot_scalability(opt_nproc)
        component.plot_scalability_n(opt_nproc)
        nproc_acc += opt_nproc
        chpsy_acc += component.get_chpsy(opt_nproc)

    # We have to add the cpl_cost to the CHPSY
    chpsy_acc += optimal_result['cpl_chpsy']

    if num_components > 1:
        print("\n -------------------------------\n")
        print("Total number of processes: %i" % nproc_acc)
        print("Expected coupled CHPSY: %i" % chpsy_acc)
        print("Expected coupled SYPD: %.2f" % optimal_result['SYPD'])
        print("Expected coupling cost: %.2f %%, %.2f (CHPSY)" % (optimal_result['cpl_cost']*100, optimal_result['cpl_chpsy']))
        print("%s/%s speed ratio: %.2f" % (list_components_class_interpolated[0].name, list_components_class_interpolated[1].name, optimal_result['speed_ratio']))
        print("Coupled Objective Function: %.3f" % optimal_result['objective_f'])

    fig, ax1 = plt.subplots()
    legend = list()
    for i in range(num_components):
        c = list_components_class_interpolated[i]
        c.fitness.plot(x='nproc', y='fitness', legend=True, ax=ax1)
        ax1.plot(optimal_result['nproc_' + c.name], c.get_fitness([optimal_result['nproc_' + c.name]]).fitness, 'o')
        legend.append(c.name)
        legend.append("optimal " + c.name)
    plt.title("Fitness values")
    plt.legend(legend)
    fig_name = "Fitness_values.png"
    plt.savefig("./img/" + fig_name)
    #plt.show()

    # Save top configurations as txt file
    out_file = "top_configurations.txt"
    f = open(out_file, "w")
    f.write(list_components_class_interpolated[0].name + '_nprocs=( ' + ''.join('%s ' % x[0] for x in optimal_result['top_configurations']) + ')\n')
    f.write(list_components_class_interpolated[1].name + '_nprocs=( ' + ''.join('%s ' % x[1] for x in optimal_result['top_configurations']) + ')\n')


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("No configuration file provided!")
        exit(1)

    # Create the directory to save the plots
    os.makedirs("./img/", exist_ok=True)

    config_file = open(sys.argv[1])
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Load global parameters form the YAML config file
    num_components = len(config['Components'])
    TTS_r = config['General']['TTS_ratio']
    ETS_r = 1 - TTS_r
    nproc_step = config['General']['nproc_step']
    method = config['General']['interpo_method']  # ['linear', 'slinear', 'quadratic', 'cubic']
    max_nproc = config['General']['max_nproc']
    show_plots = config['General']['show_plots']

    from component_class import Component
    list_components_scalability_df = list()
    list_components_class = list()
    list_components_interpolated = list()
    list_components_class_interpolated = list()

    information_per_ts_provided = True
    # Load component data
    for component in config['Components']:
        if component['timestep_info'] != None:
            ts_info_df = pd.read_csv(component['timestep_info'])
            ts_info_df.rename(columns={ts_info_df.columns[0]: "ts_id"}, inplace=True)
            ts_info_nproc = component['timestep_nproc']
        else:  # No information per ts provided. We pass an empty df and assume regular ts lengths
            ts_info_df = pd.DataFrame()
            ts_info_nproc = 0
            information_per_ts_provided = False

        component_df = pd.read_csv(component['file'])
        component_df = component_df[component_df.nproc <= max_nproc]
        list_components_scalability_df.append(component_df)  # Save this just for debugging
        # Make sure that the nproc restriction is inside the range of scalability provided for that component
        if component['nproc_restriction'] != None:
            component['nproc_restriction'] = [np for np in component['nproc_restriction'] if
                                              np <= component_df.nproc.max() and np >= component_df.nproc.min()]
            print("Component %s has a nproc restriction:" % component['name'])
            print("[", *component['nproc_restriction'], "]")

        component_class = Component(component['name'], component_df.nproc, component_df.SYPD,
                                    component['nproc_restriction'], ts_info_df, component['timestep_nproc'], TTS_r, ETS_r)
        list_components_class.append(component_class)
        # Interpolate the data
        df_component_interpolated = interpolate_data(component_class, nproc_step)
        list_components_interpolated.append(df_component_interpolated)

        # TODO: Select one of the methods
        comp_interpolated = pd.DataFrame({'nproc': df_component_interpolated.nproc,
                                           'SYPD': df_component_interpolated[method]})

        c1_n = Component(component['name'], comp_interpolated.nproc, comp_interpolated.SYPD,
                         component['nproc_restriction'], ts_info_df, ts_info_nproc, TTS_r, ETS_r)
        list_components_class_interpolated.append(c1_n)

    if show_plots:
        check_interpo(num_components, list_components_class_interpolated, list_components_interpolated)

    from brute_force import brute_force, new_brute_force
    optimal_result = new_brute_force(num_components, list_components_class_interpolated, max_nproc, show_plots)

    print_result(num_components, list_components_class_interpolated, optimal_result)
