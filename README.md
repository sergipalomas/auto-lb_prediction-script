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
  nproc_restriction:
  # [48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624,672, 720, 768, 816, 864, 912, 960, 1008 ]
  timestep_info:
  timestep_nproc:

- name: comp2
  file: data/comp2.csv
  nproc_restriction:
  timestep_info: 
  timestep_nproc:

#- name: comp3
#  file: data/comp3.csv
#  nproc_restriction:
#  timestep_info:
#  timestep_nproc:

General:
  max_nproc: 2000
  TTS_ratio: .5
  interpo_method: quadratic
  show_plots: False
  nproc_step: 24
```

- Components: List of components with their corresponding configurations.
- General: General settings including maximum processors, TTS ratio, interpolation method, etc.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

