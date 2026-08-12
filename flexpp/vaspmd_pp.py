from ase.io import read, write, Trajectory
from ase.db import connect
from flexpp.utils import default_nelect_from_potcar
from ase.parallel import world, parprint
from ase import units
import numpy as np
import os
import time


def extract_VASPsolpp(dirName=".", outcar="OUTCAR"):
    tmpHome = os.getcwd()
    os.chdir(dirName)
    with open(outcar) as f:
        lines = f.readlines()
        nelect_now = float([l for l in lines if "NELECT" in l][-1].split()[2])
        pot_now = float(
            [l for l in lines if "Fermi energy" in l][-1].split()[2]
        )  # @ renata, htis was the same as grepping the EFERMI from the opt
        pot_0 = float([l for l in lines if "EFERMI_ref" in l][-1].split()[2])
        e_gcdft = float(
            [l for l in lines if "energy  without entropy" in l][-1].split()[-1]
        )
    parprint(nelect_now, pot_now, pot_0, e_gcdft)
    os.chdir(tmpHome)
    return nelect_now, pot_now, pot_0, e_gcdft


def get_GCDFT_pp(atoms, pot_ref=4.57, dirName=".", outcar="OUTCAR", nelect_neu=None):
    if nelect_neu is None:
        nelect_neu = default_nelect_from_potcar(atoms, dirName=dirName)

    nelect_now, pot_now, pot_0, e_gcdft = extract_VASPsolpp(dirName, outcar)
    nelect_net = nelect_now - nelect_neu

    u_she = -(pot_now)  # - pot_0 ) # + u_ref
    return (
        nelect_net,
        u_she,
        e_gcdft,
    )  # to modify to return the energy of the system in the end


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
    pot_ref=-4.57,
    pot_step=1.0,  # default just takes the diff btw potential and the target
    continue_run=True,
):
    write("ini.vasp", atoms)
    nelect_neu = default_nelect_from_potcar(atoms)
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        parprint(
            f"Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)"
        )
    nelect_net_now = 0
    # traj = Trajectory('sg_atoms.traj', 'a')
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
        # GCDFT data extraction
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT_pp(
            atoms, pot_ref, dirName="."
        )
        # check how to do it
        nelect_net_new = nelect_net_now + (pot_ref - pot_now) * pot_step  # changed sign
        # Blue moon ensemble data extraction
        os.system("cat REPORT >> bluemoon_pp.txt")

        cv, lam, inv_sqrt_z, gkt, constrained_force, temp = get_bluemoon()

        geom_fin = None
        for attempt in range(5):
            try:
                geom_fin = read("vasprun.xml")
                break  # success!
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
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)=N/A T={temp:.1f}K "
                    f"efermi={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                )

        # update the NELECT in INCAR (1st line)
        with open("INCAR") as f:
            incar = f.readlines()

        # Replace NELECT line if found, else insert at top
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
