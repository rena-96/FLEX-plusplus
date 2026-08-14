import os
import time

import numpy as np
from ase import units
from ase.io import Trajectory, read, write
from ase.parallel import parprint, world

from flexpp.utils import default_nelect_from_potcar


def extract_VASPsolpp(dirName='.', outcar='OUTCAR'):
    """Read electron number, Fermi level and energy from a VASPsol++ OUTCAR.

    VASPsol++ already reports a grand-canonical energy, so unlike the VASPsol
    path no Fermi shift correction has to be applied afterwards. The last
    occurrence of each quantity in the file is used.

    Parameters
    ----------
    dirName : str
        Directory of the finished VASP calculation.
    outcar : str
        Name of the OUTCAR file inside `dirName`.

    Returns
    -------
    tuple of float
        (nelect_now, pot_now, pot_0, e_gcdft): electrons in the cell in |e|,
        the Fermi energy in eV, the reference Fermi energy EFERMI_ref in eV,
        and the grand-canonical energy without entropy in eV.
    """
    tmpHome = os.getcwd()
    os.chdir(dirName)
    with open(outcar) as f:
        lines = f.readlines()
        nelect_now = float([l for l in lines if 'NELECT' in l][-1].split()[2])
        pot_now = float([l for l in lines if 'Fermi energy' in l][-1].split()[2])
        pot_0 = float([l for l in lines if 'EFERMI_ref' in l][-1].split()[2])
        e_gcdft = float([l for l in lines if 'energy  without entropy' in l][-1].split()[-1])
    parprint(nelect_now, pot_now, pot_0, e_gcdft)
    os.chdir(tmpHome)
    return nelect_now, pot_now, pot_0,  e_gcdft


def get_GCDFT_pp(atoms, pot_ref=4.57, dirName='.', outcar='OUTCAR', nelect_neu=None):
    """Grand-canonical analysis of a finished VASPsol++ calculation.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure of the calculation, used to count the neutral electron
        number when `nelect_neu` is not given.
    pot_ref : float
        Absolute potential of the reference electrode, in V. Currently unused,
        the returned potential is on the VASPsol++ scale.
    dirName : str
        Directory of the finished VASP calculation.
    outcar : str
        Name of the OUTCAR file inside `dirName`.
    nelect_neu : float, optional
        Valence electrons of the neutral cell in |e|. Read from the POTCAR in
        `dirName` when None.

    Returns
    -------
    tuple of float
        (nelect_net, u_she, e_gcdft): excess electrons in |e|, the electrode
        potential in V, and the grand-canonical energy in eV.
    """
    if nelect_neu is None:
        nelect_neu = default_nelect_from_potcar(atoms, dirName=dirName)

    nelect_now, pot_now, pot_0,  e_gcdft = extract_VASPsolpp(dirName, outcar)
    nelect_net = nelect_now - nelect_neu

    u_she = - (pot_now)
    return nelect_net, u_she, e_gcdft


def get_bluemoon(filename="REPORT"):
    """
    Parses a single-frame REPORT file from VASP Blue Moon sampling.

    Returns:
        Tuple of:
        (cv, lambda, |z|^(-1/2), GkT, |z|^(-1/2)*(lambda+GkT), T_inst)
    """
    cv = lam = inv_sqrt_z = gkt = corrected_force = temp = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("cc> S") or line.startswith("cc> R") or line.startswith("cc> D"):
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
    nsteps = 10,
    pot_target=-4.57,
    pot_ref=-4.57,
    pot_step=None,
    timestep_fs = 1.,
    capacitance = 1.,
    R = 100,
    temperature_K = 300,
    continue_run=True
):
    """Drive a VASPsol++ slow-growth run, updating NELECT with an FDT thermostat.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure of the run, used to count the neutral electron number and
        written out as ini.vasp for reference.
    nsteps : int
        Number of VASP invocations, i.e. slow-growth steps.
    pot_target : float
        Target potential as an absolute value, in V.
    pot_ref : float
        Absolute potential of the reference electrode, in V.
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
        Resume from the existing sg_atoms_pp.traj, skipping the steps already
        stored there, instead of counting from zero.
    """
    write('ini.vasp', atoms)
    nelect_neu = default_nelect_from_potcar(atoms)
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        parprint(f'Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)')
    nelect_net_now = 0
    traj_file = 'sg_atoms_pp.traj'
    if os.path.exists(traj_file) and continue_run:
        traj = Trajectory(traj_file, 'a')
        steps_completed = len(traj)
        parprint(f"Continuing from step {steps_completed}/{nsteps}")
    else:
        traj = Trajectory(traj_file, 'a')
        steps_completed = 0
        parprint("Starting a new run")
    for i in range(steps_completed, nsteps):
        os.system('$VASP_COMMAND > vasp.out')
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT_pp(atoms, pot_ref, dirName='.')
        print('nelect_net_now, pot_now, energy_gcdft_now', nelect_net_now, pot_now, energy_gcdft_now)
        rn = np.zeros(1)
        if world.rank == 0:
            rn[0] = np.random.standard_normal()
        world.broadcast(rn, 0)
        nelect_net_new = (
            nelect_net_now
            + capacitance * (pot_now - pot_target)
            * (1 - np.exp(-timestep_fs / (R*capacitance)))
            + rn[0] * np.sqrt(
                units.kB * temperature_K * capacitance
                * (1 - np.exp(-2 * timestep_fs / (R*capacitance))))
        )
        os.system('cat REPORT >> bluemoon_pp.txt')

        cv, lam, inv_sqrt_z, gkt, constrained_force, temp = get_bluemoon()
        geom_fin = None
        for attempt in range(5):
            try:
                geom_fin = read('vasprun.xml')
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
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)=N/A T={temp:.1f}K "
                    f"efermi={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                    f"fe_grad=N/A\n"
                )

        with open('INCAR') as f:
            incar = f.readlines()

        found = False
        for i, line in enumerate(incar):
            if line.strip().startswith('NELECT'):
                incar[i] = f'NELECT={nelect_net_new + nelect_neu}\n'
                found = True
                break
        if not found:
            incar.insert(0, f'NELECT={nelect_net_new + nelect_neu}\n')

        with open('INCAR', 'w') as f:
            f.writelines(incar)
        os.system('cp CONTCAR POSCAR')
