import os
import time

import numpy as np
from ase.calculators.vasp import Vasp
from ase.geometry import find_mic
from ase.io import read
from ase.parallel import parprint
from flexpp.data import pp_dict



def get_nelect_neu(poscar='POSCAR', potcar='POTCAR', dirName='.'):
    """Count the valence electrons of the charge-neutral cell from VASP input files.

    The valence charges (ZVAL) are taken from the POTCAR and matched to the
    chemical symbols found in the POSCAR.

    Parameters
    ----------
    poscar : str
        Name of the structure file inside `dirName`.
    potcar : str
        Name of the pseudopotential file inside `dirName`.
    dirName : str
        Directory holding both files.

    Returns
    -------
    float
        Total number of valence electrons of the neutral system, in |e|.
    """
    elem_list = [l.split()[-2]
                 for l in open(f'{dirName}/{potcar}').readlines() if 'TITEL' in l]
    elem_list = [e.split('_')[0] if '_' in e else e for e in elem_list]
    zval_list = [eval(l.split()[-4])
                 for l in open(f'{dirName}/{potcar}').readlines() if 'ZVAL' in l]
    atom_list = read(f'{dirName}/{poscar}').get_chemical_symbols()
    return sum([zval_list[elem_list.index(a)] for a in atom_list])


def get_nelect_neu_ase(atoms):
    """Count the valence electrons of the neutral cell using the attached VASP calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure with an attached `ase.calculators.vasp.Vasp` calculator.

    Returns
    -------
    float
        Total number of valence electrons of the neutral system, in |e|.
    """
    calc_tmp = Vasp()
    calc_tmp.fromdict(atoms.calc.todict())
    calc_tmp.set(directory='.', charge=0)
    calc_tmp.write_input(atoms)
    nelect_neu = get_nelect_neu()
    for f in ['POSCAR', 'POTCAR', 'INCAR', 'KPOINTS', 'ase-sort.dat']:
        os.remove(f)
    return nelect_neu


def get_nelect_dict_pp(atoms, setting='recommended', pp_dir='potpaw_PBE'):
    """Map each element in `atoms` to its ZVAL from the pseudopotential library.

    The POTCAR of every element is looked up under
    $VASP_PP_PATH/`pp_dir`/<element><suffix>, where the suffix is taken from
    `flexpp.data.pp_dict` for the chosen `setting`.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose chemical symbols define the elements to look up.
    setting : str
        Key of `pp_dict` ('minimal', 'recommended', 'materialsproject', 'gw').
    pp_dir : str
        Pseudopotential family directory inside $VASP_PP_PATH.

    Returns
    -------
    dict
        Mapping {chemical symbol: valence electrons per atom}.
    """
    list_elems_unique = list(set(atoms.get_chemical_symbols()))
    vasp_pp_path = os.environ['VASP_PP_PATH']
    nelect_dict = {}
    for elem in list_elems_unique:
        elem_pp = elem
        try:
            elem_pp += pp_dict[setting][elem]
        except:
            pass
        nelect_dict[elem] = eval([l for l in open(f'{vasp_pp_path}/{pp_dir}/{elem_pp}/POTCAR').readlines() if 'ZVAL' in l][0].split()[5])
    return nelect_dict


def get_nelect_neu_pp(atoms, setting='recommended', pp_dir='potpaw_PBE'):
    """Sum the valence electrons of the neutral cell from the pseudopotential library.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to count the electrons of.
    setting : str
        Key of `pp_dict` selecting the pseudopotential variants.
    pp_dir : str
        Pseudopotential family directory inside $VASP_PP_PATH.

    Returns
    -------
    float
        Total number of valence electrons of the neutral system, in |e|.
    """
    list_elems = atoms.get_chemical_symbols()
    nelect_dict = get_nelect_dict_pp(atoms=atoms, setting=setting, pp_dir=pp_dir)
    nelect_neu = 0
    for elem in list_elems:
        nelect_neu += nelect_dict[elem]
    return nelect_neu


def get_nelect_incar(workdir):
    """Read the NELECT value set in the INCAR of a run directory.

    Parameters
    ----------
    workdir : str
        Directory containing the INCAR.

    Returns
    -------
    float
        The number of electrons requested by the INCAR, in |e|.
    """
    return eval([l for l in open(f'{workdir}/INCAR').readlines() if 'NELECT' in l][0].split()[-1])


def extract_VASPsol(dirName='.', outname='vasp.out'):
    """Collect the quantities needed for a grand-canonical energy from a VASPsol run.

    Parameters
    ----------
    dirName : str
        Directory of the finished VASP calculation.
    outname : str
        Name of the VASP stdout file holding the FERMI_SHIFT line.

    Returns
    -------
    tuple of float
        (nelect_now, energy_dft, e_fermi, fermi_shift): electrons in the cell
        in |e|, the DFT free energy in eV (from OSZICAR), the Fermi level in
        eV (from OUTCAR) and the VASPsol Fermi shift in eV.
    """
    tmpHome = os.getcwd()
    os.chdir(dirName)
    fermi_shift = eval([l for l in open(outname).readlines()
                        if 'FERMI_SHIFT' in l][-1].split()[2])
    e_fermi = eval([l for l in open('OUTCAR').readlines()
                    if 'E-fermi' in l][-1].split()[2])
    nelect_now = eval([l for l in open('OUTCAR').readlines()
                       if 'NELECT' in l][-1].split()[2])
    energy_dft = eval([l for l in open('OSZICAR').readlines()
                       if 'F=' in l][-1].split()[4])
    os.chdir(tmpHome)
    return nelect_now, energy_dft, e_fermi, fermi_shift


def get_GCDFT(u_ref=4.6, poscar='POSCAR', potcar='POTCAR', dirName='.', outname='vasp.out'):
    """Grand-canonical analysis of a finished VASPsol calculation.

    The work function is obtained as -(E_fermi + fermi_shift), the potential
    vs. SHE by subtracting `u_ref`, and the grand-canonical energy by adding
    the electron-reservoir term nelect_net * work_function to the DFT energy.
    Reading is retried once after 5 s, since the output files may still be
    flushed when this is called right after VASP exits.

    Parameters
    ----------
    u_ref : float
        Absolute potential of the SHE reference, in V.
    poscar, potcar : str
        Structure and pseudopotential file names used to count the neutral
        electron number.
    dirName : str
        Directory of the finished VASP calculation.
    outname : str
        Name of the VASP stdout file.

    Returns
    -------
    tuple of float
        (nelect_net, u_she, fermi_shift, energy_gcdft): excess electrons in
        |e|, the electrode potential vs. SHE in V, the VASPsol Fermi shift in
        eV, and the grand-canonical energy in eV.
    """

    try:
        nelect_now, energy_dft, e_fermi, fermi_shift = extract_VASPsol(dirName, outname)
    except Exception as e:
        print(f"Warning: Failed to read OUTCAR, retrying in 5 seconds...\nError: {e}")
        time.sleep(5)
        nelect_now, energy_dft, e_fermi, fermi_shift = extract_VASPsol(dirName, outname)
    nelect_neu = get_nelect_neu(poscar, potcar, dirName)
    nelect_net = nelect_now - nelect_neu
    work_function = - e_fermi - fermi_shift
    u_she = work_function - u_ref
    energy_gcdft = energy_dft + nelect_net * work_function
    return nelect_net, u_she, fermi_shift, energy_gcdft




def read_potcar_numbers_of_electrons(dirName='.', file_or_filename='POTCAR'):
    """Read the (symbol, ZVAL) pair of every block of a POTCAR file.

    Parameters
    ----------
    dirName : str
        Directory holding the POTCAR. Ignored when `file_or_filename` is an
        already-open file object.
    file_or_filename : str or file object
        Name of the POTCAR inside `dirName`, or an open file object to read
        from. A file object passed in is left open for the caller to close.

    Returns
    -------
    list of tuple
        (chemical symbol, valence electrons per atom in |e|) for each block.
    """
    close_after = False
    if isinstance(file_or_filename, str):
        path = os.path.join(dirName, file_or_filename)
        f = open(path, 'r')
        close_after = True
    else:
        f = file_or_filename

    try:
        lines = f.readlines()
    finally:
        if close_after:
            f.close()

    nelect = []
    for n, line in enumerate(lines):
        if 'TITEL' in line:
            try:
                symbol = line.split('=')[1].split()[1].split('_')[0].strip()
                for i in range(n, n + 10):
                    if i < len(lines) and 'ZVAL' in lines[i]:
                        valence = float(lines[i].split('=')[-1].strip().split()[0])
                        nelect.append((symbol, valence))
                        break
            except Exception as e:
                raise ValueError(f"Failed to parse POTCAR near line {n}: {e}")

    return nelect


def default_nelect_from_potcar(atoms, dirName='.', potcar='POTCAR'):
    """Count the valence electrons of the neutral cell from a POTCAR.

    Parameters
    ----------
    atoms : ase.Atoms or list of ase.Atoms
        Structure to count the electrons of. When a list (e.g. a trajectory)
        is given, only its first frame is used; the composition is assumed
        constant along the trajectory.
    dirName : str
        Directory holding the POTCAR.
    potcar : str
        Name of the pseudopotential file inside `dirName`.

    Returns
    -------
    float
        Total number of valence electrons of the neutral system, in |e|.
    """
    if isinstance(atoms, list):
        if len(atoms) == 0:
            raise ValueError("Empty trajectory provided.")
        atoms = atoms[0]

    symbol_count = {}
    for symbol in atoms.get_chemical_symbols():
        symbol_count[symbol] = symbol_count.get(symbol, 0) + 1

    symbol_valences = read_potcar_numbers_of_electrons(dirName, potcar)

    default_nelect = 0
    for symbol, valence in symbol_valences:
        if symbol not in symbol_count:
            raise ValueError(f"Symbol {symbol} in POTCAR not found in atoms.")
        count = symbol_count[symbol]
        default_nelect += count * valence

    return default_nelect



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


def get_GCDFT_pp(atoms, pot_ref=4.07, dirName='.', outcar='OUTCAR', nelect_neu=None):
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

