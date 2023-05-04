#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.

"""

import signac
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    ''''''
    parameters = OrderedDict()

    ### SYSTEM GENERATION PARAMETERS ###
    parameters["density"] = [1.0]
    parameters["chain_lengths"] = [15]
    parameters["n_compounds"] = [60]
    parameters["remove_hydrogens"] = [
            #True,
            False
    ]
    parameters["remove_charges"] = [
            #True,
            False
    ]

    ### SIMULATION PARAMETERS ###
    parameters["tau_kt"] = [0.1]
    parameters["tau_pressure"] = [0.1]
    parameters["dt"] = [0.0001]
    parameters["r_cut"] = [2.5]
    parameters["sim_seed"] = [42]
    parameters["shrink_steps"] = [3e7]
    parameters["shrink_period"] = [100000]
    parameters["shrink_kT"] = [8.0]
    parameters["gsd_write_freq"] = [200000]
    parameters["log_write_freq"] = [10000]

    ### Quench related parameters ###
    parameters["kT"] = [1.4]
    parameters["pressure"] = [
            0.015, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0
    ]
    parameters["n_steps"] = [1e7]
    parameters["extra_steps"] = [5e6]
    parameters["neff_samples"] = [5000]
    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project("pps") # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)

    project.write_statepoints()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
