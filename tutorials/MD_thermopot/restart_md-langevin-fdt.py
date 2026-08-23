from ase.calculators.vasp import Vasp
from ase.io import read, Trajectory
from md_fdt_restart_pp import langevin_fdt
import sys
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

target = 4.27
vaspsol = Vasp(
    lcharg=False,
    npar=4,
    xc="pbe",
    ivdw=11,
    ismear=0,
    sigma=0.1,
    encut=300,
    lreal="Auto",
    prec="High",
    # isym=0,
    algo="Fast",
    nelm=300,
    ediff=1e-6,
    gamma=True,
    kpts=[3, 3, 1],
    #    charge=0,
    efermi_ref=-target,
    lsol=True,
    eb_k=78.4,
    tau=0,
    lambda_d_k=3,
    setups="recommended",
    txt="vasp.out",
    directory="tmp",
)

geom_inp = read("./md-fdt.traj", index="-1")
# geom_inp.pbc = True
# MaxwellBoltzmannDistribution(geom_inp, temperature_K=300)
langevin_fdt(
    atoms=geom_inp,
    timestep_fs=0.4,
    fix_com=False,
    temperature_K=300,
    capacitance=0.1,
    R=10,  # was 1000
    nsteps=10000,
    # friction=0.1,
    friction=0.01,
    label="md-fdt",
    pot_target=target,
    pot_step=-0.1,
    potentiostat_nsteps=1,  # was 5
    calculator=vaspsol,
)
