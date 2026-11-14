# The structural connectivity of the engines, from the HCP

These scripts make the dataset
https://gitlab.inria.fr/dwasserm/neurolang-hcp-connectivity (see its
README for what it holds and how to acknowledge it), whose
`tsv/schaefer100_17.tsv.gz` and `tsv/destrieux.tsv.gz` the engines
download (`relations` of `mni_atlases`, at a tag, with their md5), from the public
high-resolution structural connectomes of Mansour et al. (2021): one
59,412 × 59,412 sparse matrix per subject of the HCP Young Adult study
(1,064 of them, 557 GB), in the authors' bucket
(https://github.com/sina-mansour/neural-identity).

On a Slurm cluster, with a Python of numpy, scipy, nibabel, nilearn and
pandas in `WORK/.venv`, and this repository in `WORK/NeuroLang`:

1. `python atlases.py WORK` (on a node with the internet): the parcel of
   each of the 59,412 vertices in Schaefer 2018 (100 to 1000 parcels, 7
   and 17 networks), Glasser 2016 and Destrieux 2009, in `WORK/atlases.npz`.
2. `sbatch --array=1-3 parcellate.sbatch WORK STAGING OUT`: workers that
   reduce each connectome brought into `STAGING` (a scratch directory) to
   the matrix of each atlas, in `OUT/<subject>.npz`, and delete it.
3. The connectomes, into `STAGING`, by one of:
   - `python download.py STAGING OUT` on the cluster's login node;
   - `relay.sh HOST STAGING OUT LOCAL_DIR` on another machine, when the
     cluster's downloads are slow: each file through `LOCAL_DIR`, copied
     with scp.
   Both skip what is done, so they resume, and end with `STAGING/DONE`,
   which ends the workers.
4. `python combine.py WORK OUT STAGING/glasser OUT_DIR`: the mean and
   standard deviation matrices of each atlas (`<atlas>_mean.parquet`,
   `<atlas>_sd.parquet`, their columns and index the regions), the
   subjects (`subjects.parquet`), a TSV per atlas, and
   `validation.tsv`, our Glasser matrices against the authors', subject by
   subject.

A new version of the dataset is a new tag there: the engines' URLs and
md5s name it.

On Margaret (2026-11): `WORK=/data/parietal/store3/work/dwasserm/hcp_sc`,
`STAGING=/scratch/dwasserm/hcp_sc/staging`; the login node downloaded at
0.1 MB/s, so the files were relayed through another machine.
