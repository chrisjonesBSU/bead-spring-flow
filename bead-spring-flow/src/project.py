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
def sampled(job):
    return job.doc.get("done")


@MyProject.label
def initialized(job):
    pass


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    import hoomd_polymers
    from hoomd_polymers.systems import Pack
    import hoomd_polymers.forcefields
    from hoomd_polymers.forcefields import OPLS_AA_PPS
    from hoomd_polymers.sim import Simulation
    from hoomd_polymers.molecules import PPS
    from cmeutils.sampling import is_equilibrated, equil_sample
    import numpy as np

    with job:
        print("------------------------------------")
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("------------------------------------")

        system = Pack(
                molecule=PPS,
                density=job.sp.density,
                n_mols=job.sp.n_compounds,
                mol_kwargs = {
                    "length": job.sp.chain_lengths,
                },
                packing_expand_factor=5
        )

        system.apply_forcefield(
                forcefield=OPLS_AA_PPS(),
                make_charge_neutral=True,
                remove_charges=job.sp.remove_charges,
                remove_hydrogens=job.sp.remove_hydrogens
        )

        job.doc.ref_distance = system.reference_distance
        job.doc.ref_distance_units = "angstrom"
        job.doc.ref_mass = system.reference_mass
        job.doc.ref_mass_units = "amu"
        job.doc.ref_energy = system.reference_energy
        job.doc.ref_energy_units = "kcal/mol"

        gsd_path = os.path.join(job.ws, "trajectory.gsd")
        log_path = os.path.join(job.ws, "sim_data.txt")

        sim = Simulation(
            initial_state=system.hoomd_snapshot,
            forcefield=system.hoomd_forcefield,
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
        sim.save_restart_gsd(job.fn("restart.gsd"))
        # Log volume
        uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
                volume[job.doc.shrink_cut:],
                threshold_fraction=0.50,
                threshold_neff=job.sp.neff_samples
        )
        np.savetxt("vol_sample_indices.txt", uncorr_indices)
        np.savetxt("volume.txt", uncorr_sample)
        # Save volume results in the job doc
        job.doc.vol_start = prod_start
        job.doc.average_vol = np.mean(uncorr_sample)
        job.doc.vol_std = np.std(uncorr_sample)
        job.doc.vol_sem = np.std(uncorr_sample)/(len(uncorr_sample)**0.5)
        # Log potential energy
        uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
                pe[job.doc.shrink_cut:],
                threshold_fraction=0.50,
                threshold_neff=job.sp.neff_samples
        )
        np.savetxt("pe_sample_indices.txt", uncorr_indices)
        np.savetxt("potential_energy.txt", uncorr_sample)
        # Save potential energy results in the job doc
        job.doc.pe_start = prod_start
        job.doc.average_pe = np.mean(uncorr_sample)
        job.doc.pe_std = np.std(uncorr_sample)
        job.doc.pe_sem = np.std(uncorr_sample)/(len(uncorr_sample)**0.5)
        # Add a few more things to the job job
        job.doc.total_mass = sim.mass.to("g")
        job.doc.total_mass_units = "g"
        job.doc.total_steps = job.sp.n_steps + (extra_runs*job.sp.extra_steps)
        job.doc.total_time = job.doc.total_steps*job.doc.real_timestep*1e-6
        job.doc.total_time_units = "ns"
        job.doc.extra_runs = extra_runs
        job.doc.box_nm = sim.box_lengths.to("nm")
        job.doc.box_cm = sim.box_lengths.to("cm")
        job.doc.done = True
        job.doc.sampled = True


if __name__ == "__main__":
    MyProject().main()
