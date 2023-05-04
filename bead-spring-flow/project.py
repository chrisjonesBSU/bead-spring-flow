"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os


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


@MyProject.post(npt_done)
@MyProject.operation(directives={"ngpu": 1, "executable": "python -u"})
def NPT(job):
    import hoomd_polymers
    from hoomd_polymers.library.systems import Pack
    from hoomd_polymers.forcefields import BeadSpring
    from hoomd_polymers.sim import Simulation
    from hoomd_polymers.library.polymers import LJChain

    from cmeutils.sampling import is_equilibrated, equil_sample
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

        # Set up Forcefield:
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

        gsd_path = os.path.join(job.path, "npt_trajectory.gsd")
        log_path = os.path.join(job.path, "npt_data.txt")

        sim = Simulation(
            initial_state=snapshot,
            forcefield=bead_spring_ff.hoomd_forcefield,
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
        kT_ramp = sim.temperature_ramp(
                n_steps=job.sp.shrink_steps,
                kT_start=job.sp.shrink_kT,
                kT_final=job.sp.kT
        )
        sim.run_update_volume(
                final_box_lengths=target_box,
                n_steps=job.sp.shrink_steps,
                period=job.sp.shrink_period,
                tau_kt=job.sp.tau_kt,
                kT=kT_ramp
        )
        print("------------------------------------")
        print("Shrink step finished; running NPT...")
        print("------------------------------------")
        sim.run_NPT(
                kT=job.sp.kT,
                pressure=job.sp.pressure,
                n_steps=job.sp.n_steps,
                tau_kt=job.sp.tau_kt,
                tau_pressure=job.sp.tau_pressure
        )

        job.doc.shrink_cut = int(job.sp.shrink_steps/job.sp.log_write_freq)
        extra_runs = 0
        equilibrated = False
        while not equilibrated:
        # Open up log file, see if pressure and PE are equilibrated
            data = np.genfromtxt(job.fn("npt_data.txt"), names=True)
            volume = data["mdcomputeThermodynamicQuantitiesvolume"]
            pe = data["mdcomputeThermodynamicQuantitiespotential_energy"]
            volume_eq = is_equilibrated(
                    volume[job.doc.shrink_cut+1:],
                    threshold_neff=job.sp.neff_samples,
                    threshold_fraction=job.sp.eq_threshold,
            )[0]
            pe_eq = is_equilibrated(
                    pe[job.doc.shrink_cut:],
                    threshold_neff=job.sp.neff_samples,
                    threshold_fraction=job.sp.eq_threshold,
            )[0]
            equilibrated = all([volume_eq, pe_eq])
            print("-----------------------------------------------------")
            print(f"Not yet equilibrated. Starting run {extra_runs + 1}.")
            print("-----------------------------------------------------")
            sim.run_NPT(
                    kT=job.sp.kT,
                    pressure=job.sp.pressure,
                    n_steps=job.sp.extra_steps,
                    tau_kt=job.sp.tau_kt,
                    tau_pressure=job.sp.tau_pressure
            )
            extra_runs += 1
        print("-------------------------------------")
        print("Is equilibrated; starting sampling...")
        print("-------------------------------------")
        # Find averaged density:
        uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
                volume[job.doc.shrink_cut:],
                threshold_fraction=job.sp.eq_threshold,
                threshold_neff=job.sp.neff_samples
        )
        np.savetxt("volume.txt", uncorr_sample)
        average_volume = np.mean(uncorr_sample)
        average_box_edge = average_volume**(1/3)
        job.doc.npt_vol = average_volume
        job.doc.npt_box_edge = average_box_edge
        sim.save_restart_gsd(job.fn("restart_npt.gsd"))
        job.doc.npt_done = True
        print("Simulation complieted...")


@MyProject.pre(npt_done)
@MyProject.post(nvt_done)
@MyProject.operation(directives={"ngpu": 1, "executable": "python -u"})
def NVT(job):
    import hoomd_polymers
    from hoomd_polymers.sim import Simulation

    from cmeutils.sampling import is_equilibrated, equil_sample
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
        # Load FF from pickle
        with open(job.fn("forcefield.pickle")) as f:
            hoomd_ff = pickle.load(f)

        with gsd.hoomd.open(job.fn("restart_npt.gsd"), "rb") as traj:
            snapshot = traj[-1]

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
        # Change box to average volume from NPT run
        target_box = [job.doc.npt_box_edge]*3
        sim.run_update_volume(
                final_box_lengths=target_box,
                n_steps=2e6,
                period=1,
                tau_kt=job.sp.tau_kt,
                kT=job.sp.kT
        )
        # Run NVT
        sim.run_NVT(
                kT=job.sp.kT, n_steps=job.sp.n_steps, tau_kt=job.sp.tau_kt,
        )

        shrink_cut = int(2e6/job.sp.log_write_freq)
        extra_runs = 0
        equilibrated = False
        while not equilibrated:
        # Open up log file, see if pressure and PE are equilibrated
            data = np.genfromtxt(job.fn("nvt_data.txt"), names=True)
            pressure = data["mdcomputeThermodynamicQuantitiespressure"]
            pe = data["mdcomputeThermodynamicQuantitiespotential_energy"]
            pressure_eq = is_equilibrated(
                    pressure[shrink_cut+1:],
                    threshold_neff=job.sp.neff_samples,
                    threshold_fraction=job.sp.eq_threshold,
            )[0]
            pe_eq = is_equilibrated(
                    pe[shrink_cut:],
                    threshold_neff=job.sp.neff_samples,
                    threshold_fraction=job.sp.eq_threshold,
            )[0]
            equilibrated = all([pressure_eq, pe_eq])
            print("-----------------------------------------------------")
            print(f"Not yet equilibrated. Starting run {extra_runs + 1}.")
            print("-----------------------------------------------------")
            sim.run_NVT(
                    kT=job.sp.kT, n_steps=job.sp.extra_steps, tau_kt=job.sp.tau_kt,
            )
            extra_runs += 1
        print("-------------------------------------")
        print("Is equilibrated; starting sampling...")
        print("-------------------------------------")
        # Find averaged pressure:
        uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
                pressure[shrink_cut:],
                threshold_fraction=job.sp.eq_threshold,
                threshold_neff=job.sp.neff_samples
        )
        np.savetxt("pressure.txt", uncorr_sample)
        average_pressure = np.mean(uncorr_sample)
        pressure_std = np.std(uncorr_sample)
        job.doc.average_pressure = average_pressure
        job.doc.pressure_std = pressure_std
        # Sample PE
        uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
                pe[shrink_cut:],
                threshold_fraction=job.sp.eq_threshold,
                threshold_neff=job.sp.neff_samples
        )
        np.savetxt("potential_energy.txt", uncorr_sample)

        # Save restart.gsd
        sim.save_restart_gsd(job.fn("restart_nvt.gsd"))
        job.doc.nvt_done = True
        print("Simulation complieted...")


if __name__ == "__main__":
    MyProject().main()
