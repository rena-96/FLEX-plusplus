
from ase.io import read, write, Trajectory
from ase.db import connect
from flexpp.utils import default_nelect_from_potcar
from ase.parallel import world, parprint
from ase import units 
import numpy as np
import os
import time

def extract_VASPsolpp(dirName='.', outcar='OUTCAR'):
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
    if nelect_neu is None:
        nelect_neu = default_nelect_from_potcar(atoms, dirName=dirName)

    nelect_now, pot_now, pot_0,  e_gcdft = extract_VASPsolpp(dirName, outcar)
    nelect_net = nelect_now - nelect_neu
   
    u_she = - ( pot_now )#- pot_0 ) # + u_ref
    return nelect_net, u_she, e_gcdft #to modify to return the energy of the system in the end

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
    pot_target=-4.07,
    pot_ref=-4.07,
    pot_step=None,
    timestep_fs = 1.,
    capacitance_init = 1.,
    R = 100,
    temperature_K = 300,
    continue_run=True   
):
    write('ini.vasp', atoms)
    nelect_neu = default_nelect_from_potcar(atoms)
    if pot_step is None:
        pot_step = 0.002 * nelect_neu
        parprint(f'Potentiostat step set to {pot_step:.6f} |e|/V (NELECT_neu={nelect_neu:8.3f} |e|)')

    # initialize capacitance tracking
    capacitance = capacitance_init
    efermi_last = None
    nelect_net_last = None

    traj_file = 'sg_det_pp.traj'
    if os.path.exists(traj_file) and continue_run:
        traj = Trajectory(traj_file, 'a')
        steps_completed = len(traj)
        parprint(f"Continuing from step {steps_completed}/{nsteps}")
       # if os.path.exists('sg_det_pp.log'): #to change???
        #    with open('sg_det_pp.log', 'r') as f:
         #       lines = f.readlines()
          #      capacitance = float(lines[-1].split()[-1].split('=')[-1])
    else:
        traj = Trajectory(traj_file, 'a')
        steps_completed = 0
        parprint("Starting a new run") 

    for i in range(steps_completed, nsteps):
        os.system('$VASP_COMMAND > vasp.out')

        # GCDFT data extraction
        nelect_net_now, pot_now, energy_gcdft_now = get_GCDFT_pp(atoms, pot_ref, dirName='.')
        parprint('nelect_net_now, pot_now, energy_gcdft_now', nelect_net_now, pot_now, energy_gcdft_now)

        # update capacitance if potential has changed enough
        if efermi_last is not None and abs(pot_now - efermi_last) > 0.07:
            capacitance = (nelect_net_now - nelect_net_last) / (pot_now - efermi_last) 
            if capacitance < 0:
                capacitance = capacitance_init
            parprint(f'Updated capacitance: {capacitance:.6f}')

        # store for next step for the capacitance also in case
        efermi_last = pot_now
        nelect_net_last = nelect_net_now

        # det VASPsol++ update
        nelect_net_new = nelect_net_now + capacitance * (pot_now - pot_target)

        # Blue moon ensemble data extraction
        os.system('cat REPORT >> bluemoon_det_pp.txt')

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
        with open("sg_det_pp.log", "a") as f_log:
            if inv_sqrt_z:
                f_log.write(
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)={inv_sqrt_z:.3e} T={temp:.1f}K "
                    f"efermi={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                    f"fe_grad={constrained_force / inv_sqrt_z:.3e} capacitance={capacitance:.4f}\n"
                )
            else:
                parprint(i, cv, lam, temp)
                f_log.write(
                    f"[Step {i+1}] cv={cv:.4f} λ={lam:.3e} |z|^(-1/2)=N/A T={temp:.1f}K "
                    f"efermi={pot_now:.3f} charge_net={-nelect_net_now:.3f} gcfe={energy_gcdft_now:.3f} "
                    f"fe_grad=N/A capacitance={capacitance:.4f}\n"
                )

        # write new NELECT to INCAR
        with open('INCAR') as f:
            incar = f.readlines()
        found = False
        for j, line in enumerate(incar):
            if line.strip().startswith('NELECT'):
                incar[j] = f'NELECT = {nelect_net_new + nelect_neu:.6f}\n'
                found = True
                break
        if not found:
            incar.insert(0, f'NELECT = {nelect_net_new + nelect_neu:.6f}\n')
        with open('INCAR', 'w') as f:
            f.writelines(incar)

        os.system('cp CONTCAR POSCAR')




