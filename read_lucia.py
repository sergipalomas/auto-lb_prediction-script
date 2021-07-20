import pandas as pd

# Some parameters
# Number of cores to add to a component at each iteration
proc_step = 48
# Possible resource configurations for NEMO using elPin
elpin_cores = [49, 92, 144, 192, 229, 285, 331, 380, 411, 476, 521, 563, 605, 665, 694, 759, 806, 826, 905, 905, 1008,
               1012, 1061, 1129, 1164, 1240, 1275, 1427, 1476, 1632, 1650, 1741, 1870]

# Read data from LUCIA
file = "../data/48IFS_48NEMO_no-output_t1/table.csv"
results_df = pd.read_csv(file)
results_df.rename(columns={results_df.columns[0]: 'component'}, inplace=True)
results_df["waiting_cost"] = results_df.waiting_time * results_df.nproc

# Split components and coupled data
components_df = results_df[:-1]
coupled_df = results_df[-1:]
print(components_df, "\n", coupled_df)
# Get component names
components = components_df.component.values

# Get bottleneck component
max_waiting_cost_component = components_df.loc[components_df.waiting_cost.idxmax(), "component"]
print("Bottleneck component is %s. We increase the number of resources for this component." % max_waiting_cost_component)

# Save current experiment config.
file_hist = "./auto_model_history.txt"
history = components_df[['component', 'nproc', 'SYPD', 'CHPSY']]

# If this is the first iteration, save the performance results for each component and use them as base case to compute
# metrics like speedup and efficiency.
file_base_case = "./base_case.txt"
if not os.path.isfile(file_base_case):
    with open(file_base_case, 'a') as f:
        for i in range(components.size):
            component = components[i]
            f.write(component + " SYPD " + components_df.loc[i, "SYPD"])
            f.write(component + " CHPSY " + components_df.loc[i, "CHPSY"])

with open(file_hist, 'a') as f:
    history.to_csv(f, index=False)