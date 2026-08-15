import os
import time

from ase.io import Trajectory, read, write
from ase.parallel import parprint

from flexpp.utils import default_nelect_from_potcar, extract_VASPsolpp, get_GCDFT_pp, get_bluemoon


def slow_growth(
    atoms,
    nsteps=10,
    pot_ref=-4.07,
    pot_step=1.0,
    continue_run=True,
):
    """Drive a VASPsol++ slow-growth run with a plain proportional potentiostat.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure of the run, used to count the neutral electron number and
        written out as ini.vasp for reference.
    nsteps : int
        Number of VASP invocations, i.e. slow-growth steps.
    pot_ref : float
        Reference/target Fermi energy, in eV, that the potentiostat drives to.
    pot_step : float, optional
        Potentiostat gain in |e|/V; the default of 1.0 simply applies the raw
        potential difference. Falls back to 0.002 * NELECT_neu when None.
    continue_run : bool
        Resume from the existing sg_atoms_pp.traj, skipping the steps already
        stored there, instead of counting from zero.
    """
    write("ini.vasp", atoms)
    nelect_neu = default_nelect_from_potcar(atoms)
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        parprint(
            f"Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)"
        )
    nelect_net_now = 0
    traj_file = "sg_atoms_pp.traj"
    if os.path.exists(traj_file) and continue_run:
        traj = Trajectory(traj_file, "a")
        steps_completed = len(traj)
        parprint(f"Continuing from step {steps_completed}/{nsteps}")
    else:
        traj = Trajectory(traj_file, "a")
        steps_completed = 0
        parprint("Starting a new run")
    for i in range(steps_completed, nsteps):
        os.system("$VASP_COMMAND > vasp.out")
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT_pp(
            atoms, pot_ref, dirName="."
        )
        nelect_net_new = nelect_net_now + (pot_ref - pot_now) * pot_step
        os.system("cat REPORT >> bluemoon_pp.txt")

        cv, lam, inv_sqrt_z, gkt, constrained_force, temp = get_bluemoon()

        geom_fin = None
        for attempt in range(5):
            try:
                geom_fin = read("vasprun.xml")
                break
            except Exception as e:
                parprint(f"Read attempt {attempt+1} failed: {e}")
                time.sleep(1)
        if geom_fin is None:
            raise RuntimeError("Failed to read vasprun.xml after multiple attempts.")

        traj.write(geom_fin)
        with open("sg_pp.log", "a") as f_log:
            if inv_sqrt_z:
                f_log.write(
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)={inv_sqrt_z:.3e} T={temp:.1f}K "
                    f"efermi={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                    f"fe_grad={constrained_force / inv_sqrt_z:.3e}\n"
                )
            else:
                parprint(i, cv, lam, temp)
                f_log.write(
                    f"\n [Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)=N/A T={temp:.1f}K "
                    f"efermi={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
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
