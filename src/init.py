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
import flow
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    ''''''
    parameters = OrderedDict()

    ### SYSTEM GENERATION PARAMETERS ###
    parameters["lengths"] = [30]
    parameters["n_mols"] = [50]
    parameters["bead_sequence"] = [["A"]]
    parameters["bond_lengths"] = [{"A-A": 1.1}] # nm
    parameters["bead_mass"] = [{"A": 450}]
    parameters["density"] = [0.8]
    parameters["ref_length"] = [dict(value=1, units="nm")]
    parameters["ref_energy"] = [dict(value=1, units="kJ")]
    parameters["ref_mass"] = [dict(value=450, units="amu")]
    parameters["packing_expand_factor"] = [10]
    parameters["packing_overlap"] = [0.8]
    parameters["packing_edge"] = [0.5]

    ### FORCEFIELD INFORMATION ###
    parameters["bead_types"] = [
            [{"A": {"sigma": 1.0, "epsilon": 1.0}}]
    ]
    parameters["bond_types"] = [
            [{"A-A": {"k": 500, "r0":1.1}}]
    ]
    parameters["angle_types"] = [
            [{"A-A-A": {"k": 250, "t0":2.2}}]
    ]
    parameters["dihedral_types"] = [
            [{}]
    ]

    ### SIMULATION PARAMETERS ###
    parameters["dt"] = [0.0006]
    parameters["tau_kT"] = [100]
    parameters["tau_p"] = [1000]
    parameters["r_cut"] = [2.5]
    parameters["sim_seed"] = [42]
    parameters["shrink_steps"] = [2e6]
    parameters["shrink_period"] = [1000]
    parameters["shrink_kT"] = [6.0]
    parameters["gsd_write_freq"] = [int(5e5)]
    parameters["log_write_freq"] = [int(1e4)]
    parameters["kT"] = [6.0]
    parameters["pressure"] = [None]
    parameters["n_steps"] = [5e6]
    parameters["extra_steps"] = [1e6]
    parameters["neff_samples"] = [1]
    parameters["eq_threshold"] = [0.05]
    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project() # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("npt_done", False)
        job.doc.setdefault("nvt_done", False)
        job.doc.setdefault("pressure_eq", False)
        job.doc.setdefault("volume_eq", False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
