"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os
import pickle


class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="gpu",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpuq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )

# Definition of project-related labels (classification)
@MyProject.label
def nvt_done(job):
    return job.doc.nvt_done


@MyProject.label
def npt_done(job):
    return job.doc.npt_done

# Helpful functions
def new_system(job):
    bead_spring = LJChain(
            n_mols=job.sp.n_mols,
            lengths=job.sp.lengths,
            bead_sequence=job.sp.bead_sequence,
            bead_mass=job.sp.bead_mass,
            bond_lengths=job.sp.bond_lengths
    )

    system = Pack(
            molecules=[bead_spring.molecules],
            density=job.sp.density,
            packing_expand_factor=job.sp.packing_expand_factor,
            edge=job.sp.packing_edge,
            overlap=job.sp.packing_overlap
    )

    # Set units and create starting snapshot
    length_units = getattr(unyt, job.sp.ref_length["units"])
    ref_length = job.sp.ref_length["value"] * length_units
    job.doc.ref_length = ref_length
    job.doc.ref_length_units = job.sp.ref_length["units"]

    mass_units = getattr(unyt, job.sp.ref_mass["units"])
    ref_mass = job.sp.ref_mass["value"] * mass_units
    job.doc.ref_mass = ref_mass
    job.doc.ref_mass_units = job.sp.ref_mass["units"]

    energy_units = getattr(unyt, job.sp.ref_energy["units"])
    ref_energy = job.sp.ref_energy["value"] * energy_units
    job.doc.ref_energy = ref_energy
    job.doc.ref_energy_units = job.sp.ref_energy["units"]

    system.reference_length = ref_length
    system.reference_mass = ref_mass
    system.reference_energy = ref_energy

    snapshot = system.to_hoomd_snapshot()
    return snapshot


def new_forcefield(job):
    beads = dict()
    bonds = dict()
    angles = dict()
    dihedrals = dict()

    for bead in job.sp.bead_types[0]:
        beads[bead] = job.sp.bead_types[0][bead]

    for bond in job.sp.bond_types[0]:
        bonds[bond] = job.sp.bond_types[0][bond]

    for angle in job.sp.angle_types[0]:
        angles[angle] = job.sp.angle_types[0][angle]

    for dih in job.sp.dihedral_types[0]:
        dihedrals[dih] = job.sp.dihedral_types[0][dih]

    bead_spring_ff = BeadSpring(
            beads=beads,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
            r_cut=job.sp.r_cut
    )
    return bead_spring_ff.hoomd_forcefield


@MyProject.post(npt_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="NPT"
)
def NPT(job):
    import hoomd_polymers
    from hoomd_polymers.library.systems import Pack
    from hoomd_polymers.forcefields import BeadSpring
    from hoomd_polymers.sim import Simulation
    from hoomd_polymers.library.polymers import LJChain
    from cmeutils.signac_utils import (
            check_equilibration, sample_job, get_sample
    )
    import numpy as np
    import unyt

    with job:
        print("------------------------------------")
        print("Running NPT Simulation...")
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("------------------------------------")

        if job.isfile("restart_npt.gsd"):
            print("Restarting from NPT trajectory...")
            with gsd.hoomd.open("restart_npt.gsd") as traj:
                snapshot = traj[-1]
            pressure = job.sp.pressure
            #TODO: How to handle n_steps for restart job?
            #TODO: Make sure shrinking is finished?

        if job.isfile("restart_nvt.gsd"):
            print("Restarting from NVT trajectory...")
            with gsd.hoomd.open("restart_nvt.gsd") as traj:
                snapshot = traj[-1]
            pressure = job.doc.avg_pressure
            shrink_steps = 0
        else:
            print("Creating new system...")
            snapshot = new_system(job)
            pressure = job.sp.pressure
            shrink_steps = job.sp.shrink_steps

        if job.isfile("forcefield.pickle"):
            f = open(job.fn("forcefield.pickle"), "rb")
            hoomd_ff = pickle.load(f)
        else:
            hoomd_ff = new_forcefield(job)

        gsd_path = os.path.join(job.path, "npt_trajectory.gsd")
        log_path = os.path.join(job.path, "npt_data.txt")

        sim = Simulation(
            initial_state=snapshot,
            forcefield=hoomd_ff,
            dt=job.sp.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_file_name=log_path,
            log_write_freq=job.sp.log_write_freq
        )
        sim.pickle_forcefield(job.fn("forcefield.pickle"))

        sim.reference_length = system.reference_length
        sim.reference_mass = system.reference_mass
        sim.reference_energy = system.reference_energy

        target_box = system.target_box/job.doc.ref_length
        job.doc.target_box = target_box
        job.doc.real_timestep = sim.real_timestep.to("fs")
        job.doc.real_timestep_units = "fs"

        print("----------------------")
        print("Running shrink step...")
        print("----------------------")
        if shrink_steps:
            print("Running update volume method...")
            kT_ramp = sim.temperature_ramp(
                    n_steps=job.sp.shrink_steps,
                    kT_start=job.sp.shrink_kT,
                    kT_final=job.sp.kT
            )
            sim.run_update_volume(
                    final_box_lengths=target_box,
                    n_steps=job.sp.shrink_steps,
                    period=job.sp.shrink_period,
                    tau_kt=sim.dt*job.sp.tau_kT,
                    kT=kT_ramp
            )
            job.doc.shrink_done = True
            print("Update volume run finished...")

        print("Running NPT method...")
        sim.run_NPT(
                kT=job.sp.kT,
                pressure=pressure,
                n_steps=0,
                tau_kt=sim.dt*job.sp.tau_kT,
                tau_pressure=sim.dt*job.sp.tau_p
        )
        sim.method.thermalize_thermostat_and_barostat_dof()

        sim.run_NPT(
                kT=job.sp.kT,
                pressure=pressure,
                n_steps=job.sp.n_steps,
                tau_kt=sim.dt*job.sp.tau_kT,
                tau_pressure=sim.dt*job.sp.tau_p
        )

        shrink_cut = int(shrink_steps/job.sp.log_write_freq)
        extra_runs = 0
        equilibrated = False
        while not equilibrated:
            equilibrated = check_equilibration(
                    job=job,
                    filename="npt_data.txt",
                    variable="volume",
                    threshold_fraction=job.sp.eq_threshold,
                    threshold_neff=job.sp.neff_samples
            )
            print("-----------------------------------------------------")
            print(f"Not yet equilibrated. Starting run {extra_runs + 1}.")
            print("-----------------------------------------------------")
            sim.run_NPT(
                    kT=job.sp.kT,
                    pressure=pressure,
                    n_steps=job.sp.extra_steps,
                    tau_kt=sim.dt*job.sp.tau_kT,
                    tau_pressure=sim.dt*job.sp.tau_p
            )
            extra_runs += 1
        print("-------------------------------------")
        print("Is equilibrated; starting sampling...")
        print("-------------------------------------")
        sample_job(job=job, filename="npt_data.txt", variable="volume")
        eq_volume = get_sample(
                job=job, filename="npt_data.txt", variable="volume"
        )
        job.doc.avg_vol = np.mean(eq_volume)
        job.doc.vol_std = np.std(eq_volume)

    job.doc.npt_done = True
    print("Simulation completed...")


@MyProject.pre(npt_done)
@MyProject.post(nvt_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="NVT"
)
def NVT(job):
    import gsd.hoomd
    import hoomd_polymers
    from hoomd_polymers.sim import Simulation
    from cmeutils.signac_utils import (
            check_equilibration, sample_job, get_sample
    )
    import numpy as np
    import unyt

    with job:
        print("------------------------------------")
        print("Running NVT Simulation...")
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("------------------------------------")
        if job.isfile("restart_npt.gsd"):
            print("Restarting from NPT trajectory...")
            with gsd.hoomd.open("restart_npt.gsd") as traj:
                snapshot = traj[-1]
            target_box = [job.doc.npt_box_edge]*3
            shrink_steps = 1e6
        elif job.isfile("restart_nvt.gsd"):
            print("Restarting from NVT trajectory...")
            with gsd.hoomd.open("restart_nvt.gsd") as traj:
                snapshot = traj[-1]
            shrink_steps = 0
            #TODO: How to handle n_steps for restart job?

        else:
            print("Creating new system...")
            snapshot = new_system(job)
            target_box = system.target_box/job.doc.ref_length
            shrink_steps = job.sp.shrink_steps

        if job.isfile("forcefield.pickle"):
            f = open(job.fn("forcefield.pickle"), "rb")
            hoomd_ff = pickle.load(f)
        else:
            hoomd_ff = new_forcefield(job)

        gsd_path = os.path.join(job.path, "nvt_trajectory.gsd")
        log_path = os.path.join(job.path, "nvt_data.txt")

        sim = Simulation(
            initial_state=snapshot,
            forcefield=hoomd_ff,
            dt=job.sp.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_file_name=log_path,
            log_write_freq=job.sp.log_write_freq
        )
        sim.pickle_forcefield(job.fn("forcefield.pickle"))

        if shrink_steps:
            sim.run_update_volume(
                    final_box_lengths=target_box,
                    n_steps=shrink_steps,
                    period=1,
                    tau_kt=job.sp.dt * job.sp.tau_kT,
                    kT=job.sp.kT
            )
            job.doc.nvt_update_vol = True
        # Run NVT
        sim.run_NVT(
                kT=job.sp.kT,
                n_steps=job.sp.n_steps,
                tau_kt=job.sp.dt*job.sp.tau_kT,
        )

        shrink_cut = int(shrink_steps/job.sp.log_write_freq)
        extra_runs = 0
        equilibrated = False
        while not equilibrated:
            equilibrated = check_equilibration(
                    job=job,
                    filename="nvt_data.txt",
                    variable="potential",
                    threshold_fraction=job.sp.eq_threshold,
                    threshold_neff=job.sp.neff_samples
            )
            print("-----------------------------------------------------")
            print(f"Not yet equilibrated. Starting run {extra_runs + 1}.")
            print("-----------------------------------------------------")
            sim.run_NVT(
                    kT=job.sp.kT,
                    n_steps=job.sp.extra_steps,
                    tau_kt=sim.dt*job.sp.tau_kT,
            )
            extra_runs += 1
        print("-------------------------------------")
        print("Is equilibrated; starting sampling...")
        print("-------------------------------------")
        sample_job(job=job, filename="nvt_data.txt", variable="potential")
        eq_pe = get_sample(
                job=job, filename="nvt_data.txt", variable="potential"
        )
        job.doc.average_pe = np.mean(eq_pe)
        job.doc.pe_std = np.std(eq_pe)
        sample_job(job=job, filename="nvt_data.txt", variable="pressure")
        eq_pressure = get_sample(
                job=job, filename="nvt_data.txt", variable="pressure"
        )
        job.doc.avg_pressure = np.mean(eq_pressure)
        job.doc.std_pressure = np.std(eq_pressure)

        # Save restart.gsd
        sim.save_restart_gsd(job.fn("restart_nvt.gsd"))
        job.doc.nvt_done = True
        print("Simulation complieted...")


if __name__ == "__main__":
    MyProject().main()
