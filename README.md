hd-ms
========

Extremely fast spectrum clustering tool using Hyperdimensional Computing.

The software is available as open-source under the BSD license.

Installation
------------

_hd-ms_ requires Python 3.8+ with CUDA environment is tested on the Linux platform.

Installing _hd-ms_ is easy:

    git https://github.com/wh-xu/hd-ms.git
    sh build.sh

Running _hd-ms_ using command line
----------------

_hd-ms_ supports running using the command line and takes `MGF` peak files as input and exports the clustering result
as a comma-separated file with each MS/MS spectrum and its cluster label on a single line.

Here is an example of running _hd-ms_:

    python src/main.py ~/ms-dataset/ ./output.csv --cluster_charges 2 3 --hd_dim=1024 --hd_Q=16 --eps=0.4

This will cluster all MS/MS spectra in folder `~/ms-dataset/` and generate the `otput.csv` file. Only `Charge 2` and `Charge 3` are clustered in this configuration. The HD algorithm parameters are: `D=1024` and `Q=16` while the DBSCAN clustering threshold is `eps=0.4`.
