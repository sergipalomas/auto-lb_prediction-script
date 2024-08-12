# Prediction Script to find the best resource configuration for coupled Earth system models (ESMs)

## Description
This project contains a script designed to predict the optimal resource configuration for coupled Earth System Models (ESMs) by considering the scalability properties of each individual component. The script balances Time-to-Solution (TTS) and Energy-to-Solution (ETS) criteria based on the available parallel resources.

## Requirements
To run this script, you need the following Python packages, as listed in `requirements.txt`:

- cycler==0.10.0
- kiwisolver==1.3.1
- matplotlib==3.0.3
- numpy==1.18.3
- pandas==1.0.3
- pyparsing==2.4.7
- python-dateutil==2.8.2
- pytz==2021.3
- PyYAML==5.4
- scipy==1.5.2
- six==1.16.0

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

# Usage
To run the script, use the following command:

```bash
python main.py config.yaml
```

# Configuration file
The config.yaml file is used to set up the component-specific and general configurations. Below is an example configuration:

```yaml
---
Components:
- name: comp1
  file: data/comp1.csv
  nproc_restriction: # e.g. [ 48, 96, 144, 192, 240 ]
  timestep_info: # e.g. ts_data/comp1_ts.csv
  timestep_nproc: # e.g. 240

- name: comp2
  file: data/comp2.csv
  nproc_restriction:
  timestep_info: 
  timestep_nproc:

General:
  max_nproc: 2000
  TTS_ratio: .5
  interpo_method: quadratic
  show_plots: False
  nproc_step: 24
```

Components: List of components with their corresponding configurations
- `file`: Path to the CSV file containing the scalability curve.
- `nproc_restriction`: _(optional)_ Array specifying the allowed number of processes for the component. Example: `[48, 96, 144, 192, 240]`.
- `timestep_info`:  _(Optional)_ Path to the CSV file that contains timestep-specific information. Example: `ts_data/comp1_ts.csv`.
- `timestep_nproc`: _(Mandatory if timestep_info is used)_ Integer indicating the number of processors used in the CSV provided in `timestep_info`. Example: `240`.

General settings including maximum processors, TTS ratio, interpolation method, and other configurations. All arguments are mandatory:

- `max_nproc`: Maximum number of processors available for the simulation.
- `TTS_ratio`: A ratio between 0 and 1 that determines the weight given to Time-to-Solution (TTS). A value of `1` prioritizes speed, minimizing TTS regardless of the execution cost. The recommended value is 0.5. Note that `ETS_ratio = 1 - TTS_ratio` represents the weight for Energy-to-Solution (ETS).
- `interpo_method`: Method for interpolating scalability data. Options include `linear`, `slinear`, and `quadratic`. The recommended method is `quadratic`, but it can be adjusted if the default interpolation does not accurately reflect the component's scalability.
- `show_plots`: Boolean flag that, when set to `True`, enables the generation of debug plots. Set to `False` to disable additional plots.
- `nproc_step`: Step size for evaluating different processor configurations. A smaller step size (e.g., `1`) results in a more granular search for solutions. The step size should be chosen based on the variability and performance characteristics of the model and machine. Here, for instance, I used half of the node size of the machine (`24`).

# License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

