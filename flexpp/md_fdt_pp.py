import os

import numpy as np
from ase import units
from ase.db import connect
from ase.io import read
from ase.md.langevin import Langevin
from ase.parallel import parprint, world

from flexpp.utils import default_nelect_from_potcar, get_nelect_incar, get_nelect_neu_pp


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
        pot_now = float([l for l in lines if 'Fermi energy' in l][-1].split()[2]) #@ renata, htis was the same as grepping the EFERMI from the opt
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


def langevin_fdt(
    atoms=None,
    timestep_fs=1,
    temperature_K=300,
    nsteps = 1000,
    friction=0.1,
    label='',
    pot_target=0.0,
    pot_ref=4.6,
    pot_step=None,
    potentiostat_nsteps=10,
    calculator=None,
    capacitance =1.,
    R = 50.,
    restart_traj=True,
):
    """Run constant-potential FDT using VASPsol++.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting structure; ignored when restarting from a trajectory.
    timestep_fs : float
        MD timestep in fs, also used as the electron-dynamics timestep.
    temperature_K : float
        Temperature in K, of both the thermostat and the charge fluctuations.
    nsteps : int
        Total number of MD steps to run.
    friction : float
        Langevin friction coefficient, in ASE units.
    label : str
        Run directory and prefix for the trajectory, log and database files.
    pot_target : float
        Target electrode potential, in V.
    pot_ref : float
        Absolute potential of the reference electrode, in V.
    pot_step : float, optional
        Legacy proportional gain in |e|/V; only used for the printed report.
    potentiostat_nsteps : int
        Number of MD steps between two charge updates.
    calculator : ase.calculators.vasp.Vasp
        VASP calculator used for the forces and the VASPsol++ output.
    capacitance : float
        Electrode capacitance C, in |e|/V.
    R : float
        Circuit resistance, in fs*V/|e|, setting the relaxation time R*C.
    restart_traj : bool
        Whether a previous run may be continued; also set to True whenever
        `<label>.traj` could be read.

    Returns
    -------
    ase.Atoms
        The structure at the end of the run.
    """
    mycalc = calculator
    mycalc.set(directory=label, txt='vasp.out')
    print(f'VASP settings:\n{mycalc.todict()}')

    fn_traj = f'{label}.traj'
    try:
        geom_md = read(fn_traj)
        print(f'Reading geometry from {fn_traj}...')
        restart_traj = True
    except:
        geom_md = atoms.copy()
        geom_md.set_calculator(mycalc)
        print('Starting from input geometry...')

    if restart_traj and 'charge' in geom_md.calc.todict().keys():
        charge_now = geom_md.calc.todict()['charge']
        geom_md.set_calculator(mycalc)
        nelect_neu = get_nelect_neu_pp(geom_md)
        print(f'Charge state read from trajectory: {charge_now:.4f} |e|')
    else:
        geom_md.set_calculator(mycalc)
        nelect_neu = get_nelect_neu_pp(geom_md)
        try:
            nelect_now = get_nelect_incar(label)
            charge_now = nelect_neu-nelect_now
            print(f'Charge state read from last INCAR: {charge_now:.4f} |e|')
        except:
            charge_now = 0
            print('Starting from zero net charge...')

    dyn = Langevin(
        atoms=geom_md,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
        logfile=f'{label}.log',
        trajectory=f'{label}.traj',
    )

    print(f'\nFDT POTENTIAL BOMD (pot_target= {pot_target}, timestep={timestep_fs} fs, potentialstat per {potentiostat_nsteps} fs)')
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        print(f'Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)')

    geom_md.calc.set(charge=charge_now)
    traj_nelect = []
    traj_pot = []
    traj_gcfe = []
    nsteps_done = 0
    while nsteps_done < nsteps:
        dyn.run(potentiostat_nsteps)
        nsteps_done += potentiostat_nsteps
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT_pp(geom_md, pot_ref, dirName=label)
        traj_nelect.append(nelect_net_now)
        traj_pot.append(pot_now)
        traj_gcfe.append(energy_gcdft_now)
        np.savetxt(f'{label}/log_nelect.txt', np.array(traj_nelect))
        np.savetxt(f'{label}/log_potential.txt', np.array(traj_pot))
        np.savetxt(f'{label}/log_gcfe.txt', np.array(traj_gcfe))
        print(f'Step {len(traj_nelect):4}: NELECT_net= {traj_nelect[-1]:8.4f} |e|;  U_she= {traj_pot[-1]:8.4f} V; GCFE_el= {traj_gcfe[-1]:12.4f} eV')
        with connect(f'{label}_traj.db', append=True) as db:
            db.write(
                geom_md,
                charge_net=-traj_nelect[-1],
                nelect_net = traj_nelect[-1],
                pot=traj_pot[-1],
                gcfe=traj_gcfe[-1],
                potene=geom_md.get_potential_energy()
            )
        if nsteps_done >= nsteps:
            break
        else:
            rn = np.zeros(1)
            if world.rank == 0:
                rn[0] = np.random.standard_normal()
            world.broadcast(rn, 0)
            nelect_net_new = (
                nelect_net_now  # change sign
                + capacitance * (pot_now - pot_target)
                * (1 - np.exp(-timestep_fs / (R*capacitance)))
                + rn[0] * np.sqrt(
                    units.kB * temperature_K * capacitance
                    * (1 - np.exp(-2 * timestep_fs / (R*capacitance))))
            )
            geom_md.calc.set(charge=-nelect_net_new)
    return geom_md
