# Resolved environment for POEM-v2

The `POEM` conda env as it existed on the run machine, captured with `conda env export`
before teardown. The `config/` YAMLs describe models; this describes the interpreter and
libraries they ran under.

    conda env create -f environment/POEM.conda.yml

Captured on an NVIDIA A10G, driver 615.71.09.
