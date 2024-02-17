The recommended installation method is an editable installation after cloning or copying the repository. 
To keep dependencies clean, the use of virtual environments is encouraged. 
Here, we will demonstrate the process for [conda](https://docs.conda.io/en/latest/)/[mamba](https://mamba.readthedocs.io/en/latest/index.html) and [pip](https://pip.pypa.io/en/stable/).

See [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for detailed installation instructions for micromamba, a lightweight method to manage both python interpreters and their packages. 

For example, starting from a unix-like shell, we can download micromamba:

```sh
cd /path/to/persistent/storage/
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Maybe create a quicker alias for `micromamba`:

```sh
alias mamba=$(echo $(pwd)/micromamba/bin/micromamba)
```

Then create a new environment:

```sh
mamba env create -n opt python=3.11 pip
```

Then navigate to the main directory of `matmdl` and install in editable mode:

```sh
cd /path/to/matmdl/
pip install -e .
```

If you want to build the docs, install the additional dependencies, then build in the `documentation` folder:

```sh
pip install -e .[doc]
cd documentation
mkdocs build
```
