# The structural and functional connectivity of the engines, from the HCP

This package makes the dataset
https://gitlab.inria.fr/dwasserm/neurolang-hcp-connectivity (see its
README for what it holds and how to acknowledge it), whose tables the
NeuroLang engines download (the `relations` of `structural_connectivity`
and `functional_connectivity`, at a tag, with their md5):

- the **structural** connectivity: the high-resolution structural
  connectomes of Mansour et al. (2021), one 59,412 × 59,412 sparse matrix
  per subject of the HCP Young Adult study (1,064 of them, 557 GB), in the
  authors' public bucket (https://github.com/sina-mansour/neural-identity);
- the **functional** connectivity: the resting-state fMRI of the HCP
  subjects with the four runs of rest (ICA-FIX cleaned, MSMAll, 91,282
  grayordinates; HCP open access data, which you obtain from the HCP).

Run it from `tools/` (or with `tools/` on `PYTHONPATH`), with a Python of
numpy, scipy, pandas, pyarrow, nibabel and nilearn:

```sh
python -m hcp_connectivity atlases WORK          # needs the internet
```

writes the parcel of each of the 59,412 cortical vertices in Schaefer 2018
(100 to 1000 parcels, 7 and 17 networks), Glasser 2016 and Destrieux 2009
in `WORK/atlases.npz` (`atlases.py`); Tian's subcortex (scale 2) is mapped
onto the subcortical grayordinates of the data when they are read.

The workers sum the subjects' matrices of each atlas, and their squares,
into parts (`OUT/part-<task>.npz`, saved as they go, so that a worker
started again goes on where it stopped); `combine` makes the group's
mean and standard deviation, the engines' tables and the gradients.
`--task I --tasks N` give a worker its share; by default those of a Slurm
array, so on a cluster:

```sh
sbatch --array=0-39 --mem=12G --wrap \
  "python -m hcp_connectivity functional WORK HCP_DIR PARTS"
```

and on one machine, a loop of `--task` (in parallel as memory allows):

```sh
for i in $(seq 0 7); do
  python -m hcp_connectivity functional WORK HCP_DIR PARTS --task $i --tasks 8 &
done; wait
```

## Structural

1. Workers that parcellate the connectomes as they come into `STAGING`
   (a scratch directory: about 500 MB each, deleted once parcellated),
   any number of them, each its own `--task`:
   `python -m hcp_connectivity structural WORK STAGING PARTS --task I`.
2. The connectomes, into `STAGING`, by one of:
   - `python -m hcp_connectivity download STAGING PARTS` where the
     internet is reachable;
   - `relay.sh HOST STAGING PARTS LOCAL_DIR`, from another machine, when
     the cluster's own downloads are slow: each file through `LOCAL_DIR`,
     copied with scp (`SSH_OPTIONS` for ssh and scp).

   Both skip the subjects of the parts, and end with `STAGING/DONE`,
   which ends the workers.
3. `python -m hcp_connectivity combine structural WORK PARTS DATASET`:
   the mean and standard deviation matrices of each atlas
   (`parquet/<atlas>_mean.parquet`, `<atlas>_sd.parquet`, their columns
   and index the regions), the subjects, a TSV per atlas, the gradients,
   and `validation.tsv` (our Glasser matrices against the authors',
   subject by subject).

## Functional

1. `python -m hcp_connectivity functional WORK HCP_DIR PARTS --task I
   --tasks N`: for each subject, each run's time series averaged within
   each region (the cortical parcels of an atlas, then Tian's subcortex),
   their correlations, Fisher z, averaged over the runs.
2. `python -m hcp_connectivity combine functional WORK PARTS DATASET`:
   the correlation of the mean z (`parquet/<atlas>_functional_mean.parquet`,
   over the parcels then Tian's regions), the standard deviation of z,
   the engines' tables (two parcels, a parcel and a Tian region, two Tian
   regions) and the gradients.

## Population tractogram (the subcortex, a stop-gap)

The connectomes above end on the cortical surface: no subcortex. A
stop-gap with it is the streamline counts of ONE population-average
tractogram, between the 400 parcels of Schaefer 2018 (17 networks) and the
32 regions of Tian's subcortex (scale 2), both volume atlases of the FSL
MNI152 2 mm grid (`population.py`):

```sh
python -m hcp_connectivity population WORK OUT TRACTOGRAM... [--radius 3]
```

`TRACTOGRAM` is a `.trk`, `.trk.gz` or `.tck` file, or a folder of them
(searched recursively); `WORK` receives the atlases (needs the internet).
The tractogram used is Yeh's HCP1065 population-averaged tractography atlas
(1,065 HCP subjects, ICBM 2009a nonlinear asymmetric space, 503,085
streamlines in 87 bundle files, CC BY-SA 4.0;
https://brain.labsolver.org/hcp_trk_atlas.html, Yeh 2022,
doi:10.1038/s41467-022-32595-4), its `hcp1065_avg_tracts_trk.zip`. Each end
of a streamline is given the label of the nearest labelled voxel within
`--radius` mm (3 by default: an end often stops in the white matter); a
streamline with an end in no region is left out. It writes in `OUT`
`parquet/schaefer400_17_tian_population.parquet` (the symmetric 432 × 432
matrix, the diagonal the streamlines within a region) and the engine tables
`tsv/schaefer400_17_tian_population.tsv.gz` and `tsv/tian_population.tsv.gz`.
Caveats: they are streamline counts of that atlas (a bundle atlas: the
bundles' sampling, not the connection strengths), not means over subjects,
not of the scale of the connectomes' counts; the atlas is in ICBM 2009a
space, used as FSL MNI152 without a warp (a few millimetres at most). To be
replaced by subject-level tractography.

## Gradients

`gradients.py`: a principal component analysis of the regions'
connectivity profiles within each hemisphere, the right hemisphere's
rotated onto the left's (see its docstring);
`python -m hcp_connectivity gradients WORK DATASET` makes them again from
the matrices of a dataset.

A new version of the dataset is a new tag there: the engines' URLs and
md5s name it. The TSVs are gzipped without their time, so that a file
made again from the same data is the same.
