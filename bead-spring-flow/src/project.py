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
def done(job):
    return job.doc.get("done")


@MyProject.label
def initialized(job):
    pass


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(done)
def sample(job):
    import hoomd_polymers
    from hoomd_polymers.systems import Pack
    import hoomd_polymers.forcefields
    from hoomd_polymers.forcefields import BeadSpring 
    from hoomd_polymers.sim import Simulation
    from hoomd_polymers.polymers import LJChain 
    from cmeutils.sampling import is_equilibrated, equil_sample
    import numpy as np

    with job:
        print("------------------------------------")
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("------------------------------------")
        
        bead_spring = LJChain(
                n_mols=job.sp.n_mols,
                lengths=job.sp.lengths,
                bead_sequence=list(job.sp.bead_sequence),
                bead_mass=job.sp.bead_mass,
                bond_lengths=job.sp.bond_lengths
        )
        system = Pack(
                molecule=[bead_spring.molecules],
                density=job.sp.density,
                packing_expand_factor=job.sp.packing_expand_factor
        )
        
        # Set units and create starting snapshot
        length_units = getattr(unyt, job.sp.ref_length["units"])
        ref_length = job.sp.ref_length["value"] * length_units)
        job.doc.ref_length = ref_length

        mass_units = getattr(unyt, job.sp.ref_mass["units"])
        ref_mass = job.sp.ref_mass["value"] * mass_units)
        job.doc.ref_mass = ref_mass

        energy_units = getattr(unyt, job.sp.ref_mass["energy"])
        ref_energy = job.sp.ref_energy["value"] * energy_units)
        job.doc.ref_energy = ref_energy

        system.reference_length = ref_length
        system.reference_mass = ref_mass
        system.reference_energy = ref_energy

        snapshot = system.to_hoomd_snapshot()

        # Set up Forcefield:
        beads = dict()
        bonds = dict()
        angles = dict()
        dihedrals = dict()

        for bead in job.sp.bead_types:
            beads[bead] = job.sp.bead_types[bead]

        for bond in job.sp.bond_types:
            bonds[bond] = job.sp.bond_types[bond]

        for angle in job.sp.angle_types:
            angles[angle] = job.sp.angle_types[angle]

        for dih in job.sp.dihedral_types:
            dihedrals[dih] = job.sp.dihedral_types[dih]

        bead_spring_ff = BeadSpring(
                beads=beads,
                bonds=bonds,
                angle=angles,
                dihedrals=dihedrals,
                r_cut=job.sp.r_cut
        )
                
        gsd_path = os.path.join(job.ws, "trajectory.gsd")
        log_path = os.path.join(job.ws, "sim_data.txt")

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

        sim.reference_distance = system.reference_distance
        sim.reference_mass = system.reference_mass
        sim.reference_energy = system.reference_energy

        target_box = system.target_box*10/job.doc.ref_distance
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
            data = np.genfromtxt(job.fn("sim_data.txt"), names=True)
            volume = data["mdcomputeThermodynamicQuantitiesvolume"]
            pe = data["mdcomputeThermodynamicQuantitiespotential_energy"]
            volume_eq = is_equilibrated(
                    volume[job.doc.shrink_cut+1:],
                    threshold_neff=job.sp.neff_samples,
                    threshold_fraction=0.50,
            )[0]
            pe_eq = is_equilibrated(
                    pe[job.doc.shrink_cut:],
                    threshold_neff=job.sp.neff_samples,
                    threshold_fraction=0.50,
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


if __name__ == "__main__":
    MyProject().main()
