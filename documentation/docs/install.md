This is a small, research-oriented software package with very few total users. Therefore, the recommended installation method is an editable installation after cloning or copying the repository. 
To keep dependencies clean, the use of virtual environments is encouraged. 
Here, we will demonstrate the process for [conda](https://docs.conda.io/en/latest/)/[mamba](https://mamba.readthedocs.io/en/latest/index.html) and [pip](https://pip.pypa.io/en/stable/).

See [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for detailed installation instructions for micromamba, a lightweight method to manage both python interpreters and their packages. 

For example, starting from a unix-like shell, we can download micromamba:

```sh
cd /path/to/persistent/storage/
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Maybe create a quicker alias for the `micromamba` binary just installed or add that to a bashrc file if using:

```sh
alias mamba=$(echo $(pwd)/micromamba/bin/micromamba)
# or:
echo "alias mamba=$(echo $(pwd)/micromamba/bin/micromamba)" >> ~/.bashrc
```

Now create a new environment with the project dependencies:

```sh
cd /path/to/matmdl/
mamba create -n opt python=3.11 --file requirements.txt
```

Then activate the new environement and install this repository in editable mode:

```sh
mamba activate opt
pip install --no-build-isolation --no-deps -e .
```

If you want to build the docs, install the additional dependencies, then build in the `documentation` folder:

```sh
pip install -e '.[doc]'
cd documentation
mkdocs build
```
