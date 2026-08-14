import os

import numpy as np
from ase import units
from ase.io import Trajectory, read, write
from ase.parallel import world

from flexpp.utils import get_GCDFT, get_nelect_neu_pp


def get_bluemoon(filename="REPORT"):
    """
    Parses a single-frame REPORT file from VASP Blue Moon sampling.

    Returns:
        Tuple of:
        (cv, lambda, |z|^(-1/2), GkT, |z|^(-1/2)*(lambda+GkT), T_inst)
    """
    cv = lam = inv_sqrt_z = gkt = corrected_force = temp = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if (
                line.startswith("cc> S")
                or line.startswith("cc> R")
                or line.startswith("cc> D")
            ):
                cv = float(line.split()[2])
            elif line.startswith("b_m>"):
                parts = line.split()
                lam = float(parts[1])
                inv_sqrt_z = float(parts[2])
                gkt = float(parts[3])
                corrected_force = float(parts[4])
            elif line.startswith("tmprt>"):
                temp = float(line.split()[2])

    return cv, lam, inv_sqrt_z, gkt, corrected_force, temp


def slow_growth(
    atoms,
    nsteps=10,
    pot_target=0.0,
    pot_ref=4.6,
    pot_step=None,
    timestep_fs=1.0,
    capacitance=1.0,
    R=100,
    temperature_K=300,
    continue_run=True,
):
    """Drive a VASPsol slow-growth run, updating NELECT with an FDT thermostat.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure of the run, used to count the neutral electron number and
        written out as ini.vasp for reference.
    nsteps : int
        Number of VASP invocations, i.e. slow-growth steps.
    pot_target : float
        Target electrode potential vs. SHE, in V.
    pot_ref : float
        Absolute potential of the SHE reference, in V.
    pot_step : float, optional
        Legacy proportional gain in |e|/V. Defaults to 0.002 * NELECT_neu and
        is only reported, not used by the FDT update.
    timestep_fs : float
        Timestep of the electron dynamics, in fs.
    capacitance : float
        Electrode capacitance C, in |e|/V.
    R : float
        Circuit resistance, in fs*V/|e|, setting the relaxation time R*C.
    temperature_K : float
        Temperature in K entering the fluctuation term.
    continue_run : bool
        Resume from the existing sg_atoms.traj, skipping the steps already
        stored there, instead of counting from zero.
    """
    write("ini.vasp", atoms)
    nelect_neu = get_nelect_neu_pp(atoms)
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        print(
            f"Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)"
        )
    nelect_net_now = 0
    traj_file = "sg_atoms.traj"
    if os.path.exists(traj_file) and continue_run:
        traj = Trajectory(traj_file, "a")
        steps_completed = len(traj)
        print(f"Continuing from step {steps_completed}/{nsteps}")
    else:
        traj = Trajectory(traj_file, "a")
        steps_completed = 0
        print("Starting a new run")
    for i in range(steps_completed, nsteps):
        os.system("$VASP_COMMAND > vasp.out")
        nelect_net_now, pot_now, fermi_shift, energy_gcdft_now = get_GCDFT(
            pot_ref, dirName="."
        )
        rn = np.zeros(1)
        if world.rank == 0:
            rn[0] = np.random.standard_normal()
        world.broadcast(rn, 0)
        nelect_net_new = (
            nelect_net_now
            + capacitance
            * (pot_now - pot_target)
            * (1 - np.exp(-timestep_fs / (R * capacitance)))
            + rn[0]
            * np.sqrt(
                units.kB
                * temperature_K
                * capacitance
                * (1 - np.exp(-2 * timestep_fs / (R * capacitance)))
            )
        )
        os.system("cat REPORT >> bluemoon.txt")

        cv, lam, inv_sqrt_z, gkt, constrained_force, temp = get_bluemoon()

        geom_fin = read("vasprun.xml")
        traj.write(geom_fin)

        with open("sg.log", "a") as f_log:
            if inv_sqrt_z:
                f_log.write(
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)={inv_sqrt_z:.3e} T={temp:.1f}K "
                    f"pot={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                    f"fe_grad={constrained_force / inv_sqrt_z:.3e} shift={fermi_shift:.3f}\n"
                )
            else:
                print(i, cv, lam, temp)
                f_log.write(
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)=N/A T={temp:.1f}K "
                    f"pot={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                    f"fe_grad=N/A fermi_shift=N/A\n"
                )

        with open("INCAR") as f:
            incar = f.readlines()

        found = False
        for i, line in enumerate(incar):
            if line.strip().startswith("NELECT"):
                incar[i] = f"NELECT={nelect_net_new + nelect_neu}\n"
                found = True
                break
        if not found:
            incar.insert(0, f"NELECT={nelect_net_new + nelect_neu}\n")

        with open("INCAR", "w") as f:
            f.writelines(incar)
        os.system("cp CONTCAR POSCAR")
