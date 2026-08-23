#!/bin/bash
#SBATCH -o ./sp.out
#SBATCH -e ./sp.err
#SBATCH --partition=xeon56
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 56 # Number of MPI processes per node
#SBATCH --time=2-02:00:00


module use /home/modules/energy/modules/all/
module load ASE/3.25.0-iimkl-2023a
export ASE_VASP_COMMAND="srun vasp_std"
# module use /home/modules/energy/modules/all/
#
module load VASP/6.3.2-intel-2023a-vaspsolpp
# module load VASP/5.4.4-intel-2023a-vtst-vaspsol-wannier-beef

export PYTHONPATH=$PYTHONPATH:/home/energy/rogle/chgpot_advances
# #export ASE_VASP_VDW=/home/modules/energy/software/VASP/vasp-potpaw-5.4
# #export VASP_PP_PATH=/home/modules/energy/software/VASP/vasp-potpaw-5.4/


python ./restart_md-langevin-fdt.py
# mq submit -R p:24:xeon24el8_test:10m ./md-langevin-fdt.py

#date >> time.txt

