# FLEX++

**FL**uctuation-dissipation **EX**tension for VASPsol++

<!-- ============================================================= -->
<!-- ADD YOUR DESCRIPTION HERE                                      -->
<!-- A short paragraph on what FLEX++ does, e.g. ab initio MD,      -->
<!-- slow-growth free energetics, and applied-potential            -->
<!-- calculations built on top of VASPsol++.                       -->
<!-- ============================================================= -->

`FLEX++` is a toolkit for ... <!-- replace this line with your description -->

It is an external standalone code, interacting with VASP via the ASE interface, needless of any extra and elaborate patching on VASP.

Please make sure to cite the following if you use `FLEX++` in your research:

- ChemRxiv. [[link]](https://chemrxiv.org/) <!-- add DOI / link once available -->

## Requirements

- `Python` 3.6 or higher
- `ASE` 3.22.1 or higher
- `VASP` 5.4.1 or higher, with the `VASPsol++` add-on

## Installation

If your machine has Git installed, simply clone the repo to your local directory by:

```
git clone https://github.com/rena-96/FLEX-plusplus
```

Or, you can download and unzip the source code:

```
wget https://github.com/rena-96/FLEX-plusplus/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
mv FLEX-plusplus-master flexpp
```

After fetching the repo, add it to your `PYTHONPATH` by:

```
export PYTHONPATH=$PYTHONPATH:`pwd`/flexpp/
```

Remember to add this export line to your `~/.bashrc` or the submission script, so that the `FLEX++` package is accessible by Python when you run the job.

You need to use the absolute path (you can check it by running `pwd` in the Bash shell) for this purpose.

After these, run the following line to test:

```
python -c 'import flexpp'
```

If no error occurs, `FLEX++` should have been imported into your path!

#### ASE-related settings

In the job submission script, besides the `PYTHONPATH`, you should also set some [environment variables](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#environment-variables) for the `ASE` interface to `VASP`:

```
export PYTHONPATH=$PYTHONPATH:/xxx/flexpp
export VASPHOME=/xxx/vasp.x.x.x/bin
export VASP_COMMAND="mpirun -n $NSLOTS $VASPHOME/vasp_std"
export VASP_PP_PATH=/xxx/potentials/
```

The `VASP_COMMAND` will vary from machine to machine, depending on the job scheduler and the number of cores to use. You may need to adjust it yourself (and the script header, of course).

## Tutorials

See the `examples` folder.

## Authors

Developed by Renata Sechi, Ph.D., and Ronan Gleeson, M.Sc.

## License

Copyright © 2026 Technical University of Denmark.
