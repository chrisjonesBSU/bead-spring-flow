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
    parameters["lengths"] = [15]
    parameters["n_mols"] = [60]
    parameters["bead_sequence"] = [["A"]]
    parameters["bond_lengths"] = [{"A-A": 0.5}] # nm
    parameters["bead_mass"] = [{"A": 100}]
    parameters["density"] = [1.0]
    parameters["ref_length"] = [dict(value=1, units="nm")]
    parameters["ref_energy"] = [dict(value=1, units="kJ")]
    parameters["ref_mass"] = [dict(value=100, units="amu")]
    parameters["packing_expand_factor"] = [5]

    ### FORCEFIELD INFORMATION ###
    parameters["bead_types"] = [
            [{"A": {"sigma": 1.0, "epsilon": 1.0}}],
    ]
    parameters["bond_types"] = [
            [{"A-A": {"k": 500, "r0":0.5}}],
    ]
    parameters["angle_types"] = [
            [{"A-A-A": {"k": 100, "t0":2.2}}],
    ]
    parameters["dihedral_types"] = [
            [{}],
    ]

    ### SIMULATION PARAMETERS ###
    parameters["tau_kt"] = [0.05]
    parameters["tau_pressure"] = [0.5]
    parameters["dt"] = [0.0005]
    parameters["r_cut"] = [2.5]
    parameters["sim_seed"] = [42]
    parameters["shrink_steps"] = [5e6]
    parameters["shrink_period"] = [10000]
    parameters["shrink_kT"] = [3.0]
    parameters["gsd_write_freq"] = [int(5e4)]
    parameters["log_write_freq"] = [int(5e3)]

    ### Quench related parameters ###
    parameters["kT"] = [1.4]
    parameters["pressure"] = [None]
    parameters["n_steps"] = [1e7]
    parameters["extra_steps"] = [5e6]
    parameters["neff_samples"] = [5000]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
