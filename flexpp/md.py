import numpy as np
from ase import units
from ase.constraints import FixInternals
from ase.db import connect
from ase.io import read
from ase.md.langevin import Langevin

from flexpp.utils import get_GCDFT, get_nelect_incar, get_nelect_neu_pp


def langevin_fixpot(
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
    calculator=None
):
    """Run constant-potential Langevin BOMD with a proportional potentiostat.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting structure; ignored when restarting from a trajectory.
    timestep_fs : float
        MD timestep in fs.
    temperature_K : float
        Langevin thermostat temperature in K.
    nsteps : int
        Total number of MD steps to run.
    friction : float
        Langevin friction coefficient, in ASE units.
    label : str
        Run directory and prefix for the trajectory, log and database files.
    pot_target : float
        Target electrode potential vs. SHE, in V.
    pot_ref : float
        Absolute potential of the SHE reference, in V.
    pot_step : float, optional
        Potentiostat gain in |e|/V. Defaults to 0.002 * NELECT_neu.
    potentiostat_nsteps : int
        Number of MD steps between two potentiostat corrections.
    calculator : ase.calculators.vasp.Vasp
        VASP calculator used for the forces and the VASPsol output.

    Returns
    -------
    ase.Atoms
        The structure at the end of the run.
    """
    mycalc = calculator
    mycalc.set(directory=label, txt='vasp.out')
    print(f'VASP settings:\n{mycalc.todict()}')

    # Read geometry if trajectory file exists
    restart_traj = False
    fn_traj = f'{label}.traj'
    try:
        geom_md = read(fn_traj)
        print(f'Reading geometry from {fn_traj}...')
        restart_traj = True
    except:
        geom_md = atoms.copy()
        geom_md.set_calculator(mycalc)
        print('Starting from input geometry...')

    # Get charge state and set calculator
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

    print(f'\nFIX-POTENTIAL BOMD (pot_target= {pot_target}, timestep={timestep_fs} fs, potentialstat per {potentiostat_nsteps} fs)')
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
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT(pot_ref, dirName=label)
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
            nelect_net_new = nelect_net_now - (pot_target-pot_now)*pot_step
            geom_md.calc.set(charge=-nelect_net_new)
    return geom_md


def langevin_fixpot_slowgrowth_combobond(
    atoms=None,
    cv_combo = None,
    cv_ini = None,
    cv_fin = None,
    grwoth_per_step = 0.0001,
    timestep_fs=1,
    temperature_K=300,
    nsteps = 10000,
    friction=0.1,
    label='',
    pot_target=0.0,
    pot_ref=4.6,
    pot_step=None,
    potentiostat_nsteps=1,
    calculator=None,
):
    """Run slow-growth constant-potential BOMD along a linear-combination bond CV.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting structure; ignored when restarting from a trajectory.
    cv_combo : list
        Bond combination defining the collective variable, in the format
        expected by `ase.constraints.FixInternals` bondcombos.
    cv_ini, cv_fin : float
        Initial and final value of the collective variable, in Angstrom.
    grwoth_per_step : float
        Increment added to the collective variable per block, in Angstrom.
    timestep_fs : float
        MD timestep in fs.
    temperature_K : float
        Langevin thermostat temperature in K.
    nsteps : int
        Maximum number of MD steps; the run also stops once `cv_fin` is passed.
    friction : float
        Langevin friction coefficient, in ASE units.
    label : str
        Run directory and prefix for the trajectory, log and database files.
    pot_target : float
        Target electrode potential vs. SHE, in V.
    pot_ref : float
        Absolute potential of the SHE reference, in V.
    pot_step : float, optional
        Potentiostat gain in |e|/V. Defaults to 0.002 * NELECT_neu.
    potentiostat_nsteps : int
        Number of MD steps between two potentiostat corrections and two CV
        increments.
    calculator : ase.calculators.vasp.Vasp
        VASP calculator used for the forces and the VASPsol output.

    Returns
    -------
    ase.Atoms
        The structure at the end of the run.
    """
    mycalc = calculator
    mycalc.set(directory=label, txt='vasp.out')
    print(f'VASP settings:\n{mycalc.todict()}')

    restart_traj = False
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

    if restart_traj:
        try:
            cv_now = eval(open(f'{label}/cv_now.txt').read())
        except:
            cv_now = None

    dyn = Langevin(
        atoms=geom_md,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
        logfile=f'{label}.log',
        trajectory=f'{label}.traj',
    )

    print(f'\nFIX-POTENTIAL BOMD (pot_target= {pot_target}, timestep={timestep_fs} fs, potentialstat per {potentiostat_nsteps} fs)')
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        print(f'Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)')

    geom_md.calc.set(charge=charge_now)
    traj_nelect = []
    traj_pot = []
    traj_gcfe = []
    nsteps_done = 0
    if cv_now is not None:
        cv_now = cv_init
    while nsteps_done < nsteps and cv_now <= cv_fin:
        geom_md.set_constraint(
            FixInternals(bondcombos=[cv_now, cv_combo])
            )
        dyn.run(potentiostat_nsteps)
        cv_now += grwoth_per_step
        nsteps_done += potentiostat_nsteps
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT(pot_ref, dirName=label)
        traj_nelect.append(nelect_net_now)
        traj_pot.append(pot_now)
        traj_gcfe.append(energy_gcdft_now)
        np.savetxt(f'{label}/log_nelect.txt', np.array(traj_nelect))
        np.savetxt(f'{label}/log_potential.txt', np.array(traj_pot))
        np.savetxt(f'{label}/log_gcfe.txt', np.array(traj_gcfe))
        np.savetxt(f'{label}/cv_now.txt', cv_now - grwoth_per_step)
        print(f'Step {len(traj_nelect):4}: NELECT_net= {traj_nelect[-1]:8.4f} |e|;  U_she= {traj_pot[-1]:8.4f} V; GCFE_el= {traj_gcfe[-1]:12.4f} eV')
        with connect(f'{label}_traj.db', append=True) as db:
            db.write(
                geom_md,
                charge_net=-traj_nelect[-1],
                nelect_net = traj_nelect[-1],
                pot=traj_pot[-1],
                gcfe=traj_gcfe[-1],
                potene=geom_md.get_potential_energy(),
                cv=cv_now - grwoth_per_step,
            )
        if nsteps_done >= nsteps:
            break
        else:
            nelect_net_new = nelect_net_now - (pot_target-pot_now)*pot_step
            geom_md.calc.set(charge=-nelect_net_new)
    return geom_md
