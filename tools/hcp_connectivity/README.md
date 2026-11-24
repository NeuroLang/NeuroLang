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

## Voxel partial functional connectivity

The relation `schaefer400_voxel_partial_functional_connectivity` of the
`functional_connectivity` engine (`voxel_functional.py`): the partial
correlation of each of the 132,032 cortical voxels of Schaefer's 400-parcel
atlas (17 networks, 2 mm) with each parcel, controlling for the other 399.
It reads the HCP's *volumetric* ICA-FIX runs
(`MNINonLinear/Results/rfMRI_REST{1,2}_{LR,RL}/*_hp2000_clean.nii.gz`),
which are on the atlas's 2 mm FSL MNI152 grid, and, from `HCP_DIR`,
`hcp_restricted.csv` (`Family_ID`, `Age_in_Yrs`) and `hcp_behavioral.csv`
(`Gender`).

1. `python -m hcp_connectivity voxel-functional WORK HCP_DIR PARTS --task I
   --tasks N`: a task of the subjects (those with the four runs and a
   phenotype); embarrassingly parallel, so on a Slurm array
   `sbatch --array=0-99 --cpus-per-task=8 --wrap "python -m
   hcp_connectivity voxel-functional WORK HCP_DIR PARTS"`. A task peaks at
   14 GB (a run is 4.3 GB as float32); on a cluster that schedules by cores
   alone (Slurm's `CR_CORE`, where `--mem` is not enforced), ask for enough
   cores that a node's tasks fit in its memory: 8 on margaret's nodes of
   2 GB a core, whose 4-core tasks ran out of memory. For each run, the voxels' series and the mean of
   each parcel's voxels (a voxel's own parcel without it); the partial
   correlation of a voxel with a parcel follows from the inverse of the
   parcels' covariance, shrunk by Ledoit-Wolf's coefficient (one per run,
   of the parcels' correlations), by a rank-2 update for each voxel
   (Woodbury), so that no matrix is inverted per voxel: 2 s a run, the
   reading of the file takes longer. nilearn's
   `ConnectivityMeasure(kind="partial correlation",
   cov_estimator=LedoitWolf())` on the standardised series of a voxel and
   the parcels gives the same values (1e-4 on a real run; it takes 2.6 s a
   voxel). The Fisher z are averaged over the four runs; a voxel without
   signal in a run (outside the subject's brain mask: about 3 %) has none
   for the subject, and a run in which a whole parcel has none is left out
   (a subject with no such run is left out; the log says which). A part (`PARTS/part-<task>.npz`, about 630 MB, float32;
   saved after each subject, so a task started again goes on) keeps the sums
   of the group's regression, below.
2. `python -m hcp_connectivity combine voxel-functional WORK PARTS DATASET
   [--threshold 0.01]`: the group, for each voxel, the regression of the z of
   the subjects with signal in it (at least 4) on their age and sex, each
   family counting once (a subject weighs 1 / the subjects of its family,
   fixed by all the eligible subjects), read at the mean age and an equal
   mix of the sexes, turned back by tanh:
   `parquet/schaefer400_17_voxel_partial_functional_mean.parquet` (the
   matrix, indexed by the voxel's centre `x y z` in MNI millimetres) and
   the engine's table
   `parquet/schaefer400_17_voxel_partial_functional_connectivity.parquet`
   (`x y z label r`, float32, the pairs whose |r| is above the threshold,
   the anticorrelations too, sorted).
   The subjects, with their family, weight, exact age and sex, go to
   `PARTS/subjects.parquet`, not to the dataset: they are HCP restricted
   data, which the published dataset must not hold (the parts too hold
   them, and stay where they are made).
   Partial correlations are small, so the number of rows above 0.02,
   0.03, 0.05, 0.1, ... (and below their negatives) is printed to choose
   the threshold: the engine's table is a Parquet file because the full
   one (53 million pairs) is too large to load. `combine` holds two parts
   and the matrices at once: 62 GB at its peak for 132,032 voxels.

   The dataset's v2.5 is that of the 966 subjects with the four runs (426
   families; 101107's REST1_LR left out, a parcel outside its brain mask),
   with `--threshold 0.01`: of 52.8 million pairs, the full matrix (194 MB)
   and 237,073 pairs whose |r| is above 0.01 (1.6 MB). A voxel and its own
   parcel are 0.02 to 0.09 (the 5th and 95th percentiles; 131,454 pairs
   above 0.01), its own parcel being its strongest for 91 % of the voxels;
   a voxel and another parcel are 0 at the median, 103,903 pairs above
   0.01, 3,068 above 0.05, at most 0.13, half of those above 0.01 within
   the voxel's own network (6 % by chance); 1,716 pairs are below -0.01,
   none below -0.031. The negative side of the other parcels, mirrored,
   puts the pairs above 0.01 that chance alone would give at about 1.7 %.

## Population tractogram (the subcortex, a stop-gap, superseded)

Superseded in the engines by the subjects' own tractography (below,
`Tractography`; the dataset's tag v2.2); kept to remake the tables of v2.0.

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
space, used as FSL MNI152 without a warp (a few millimetres at most).

## A dense tractogram (Disconnectome7T)

The tractogram of the engines `neurosynth_disconnectome7t` and
`neurostore_disconnectome7t`: Disconnectome7T (https://osf.io/5zqwg/,
Thiebaut de Schotten et al. 2020, doi:10.1038/s41467-020-18920-9), the
tractography of 180 HCP subjects' 7 T diffusion MRI, one `.trk` each
(about 1 million streamlines, 134 GB in all).

1. `python -m hcp_connectivity osf 5zqwg TRK_DIR --jobs 8` (`osf.py`):
   the files of the OSF project, checked by their md5, resumable.
2. `python -m hcp_connectivity check-space TRACTOGRAM HCP_DIR`
   (`space.py`): the streamlines against the white matter of their
   subject's own segmentation in the HCP's MNI152 space
   (`MNINonLinear/wmparc.nii.gz`), as they are, left and right swapped and
   shifted. Disconnectome7T's are the right way round, but 1 to 2 mm off
   (Karolis et al. 2019's registration): a shift of about (0, 1, 2) mm
   raises their correlation with the white matter, and differs by region.
3. `python -m hcp_connectivity fit-space HCP_DIR AFFINE TRACTOGRAM...`:
   the affine maximising that correlation (of the density of the points,
   trilinear, with the fraction of white matter, on blocks of 1.4 mm).
   Fitted on 100610, 102311 and 102816, the three agree (a stretch of 2 to
   4 % in x, 1 mm forward, 1.2 mm up); their mean,
   `disconnectome7t_to_mni152.txt`, raises the correlation of four other
   subjects by 0.016 to 0.023 and the fraction of their points in the
   white matter by 0.7 to 1.2 points.
4. `python -m hcp_connectivity dense TRK_DIR HCP_DIR OUT --subjects 20
   --affine hcp_connectivity/disconnectome7t_to_mni152.txt --count 300000
   --brain` (`dense.py`): the streamlines of at least 30 mm of the first 20
   subjects, mapped, compressed within 0.5 mm (dipy), pooled (7,570,429),
   shuffled; with `--brain` only those whose points are all within the
   MNI152 brain mask (nilearn's; 6,835,647: the 7 T data are distorted
   over the orbits and under the cerebellum, where 8 to 16 % of the points
   of a subject's streamlines leave its own brain, against 1.3 to 1.7 %
   elsewhere); the first `--count` of them, within `--megabytes`:
   `disconnectome7t.tck` (300,000 streamlines, 44 MB); and
   `disconnectome7t_wmparc.nii.gz`, the label most of the 20 subjects'
   segmentations give each voxel. Needs dipy and nilearn.

The size is set by what the engine can query, not by the disk: on a laptop
(pandas backend), all 58 bundles of freesurfer_queries at once take 12 min
and 36 GB on 300,000 streamlines; on all 3.2 million that fit in 480 MB, one
bundle takes 52 s (cst) to 11 min (af) and 20 GB, and all of them at once
do not fit in 64 GB.

## Tractography

The structural connectivity of each subject from its own tractography
(`tractography.py`), over Schaefer's 400 parcels (17 networks) and the 32
regions of Tian's subcortex (scale 2): 432 × 432. It needs MRtrix3 (3.0.4
tested), the HCP's preprocessed diffusion data in the subject's T1w space
(`T1w/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}`), its
`T1w/aparc+aseg.nii.gz` and the warp `MNINonLinear/xfms/standard2acpc_dc`;
no FSL.

1. `python -m hcp_connectivity volume-atlas WORK` (needs the internet):
   Schaefer's volume and Tian's atlas in MNI space, combined (cortex
   1-400, Tian 401-432) in `WORK/atlases`; `WORK` is also where
   `atlases.npz` (above) names the parcels.
2. `python -m hcp_connectivity tractography WORK HCP_DIR OUT --task I
   --tasks N [--limit 100] [--streamlines 5000000] [--scratch DIR]`, a task
   of the subjects that have diffusion data (the first `--limit`, sorted).
   The MRtrix3 commands are run by `$MRTRIX_CMD`: for a container,
   `MRTRIX_CMD="singularity exec mrtrix3.sif"` (with the folders of the
   data bound, `SINGULARITY_BIND`); empty, those on the `PATH`. The threads
   are `$SLURM_CPUS_PER_TASK`, or all the CPUs. For each subject, in
   `--scratch` (default `WORK/tractography`): `dwi2response dhollander`,
   `dwi2fod msmt_csd` (white matter and CSF), `5ttgen freesurfer
   -sgm_amyg_hipp`, `tckgen` (iFOD2, ACT, backtracking, cropped at the
   grey/white interface, dynamic seeding, 5 million streamlines), `tcksift2`,
   the atlas brought to the subject through the HCP's warp (nearest
   neighbour, `nibabel`/`scipy`, no FSL) and checked against its
   segmentation (Dice of Tian's structures and aseg's), and
   `tck2connectome -symmetric -zero_diagonal -assignment_radial_search 2`.
   The steps done are marked, so that a task started again goes on after
   the last one; the intermediate files, the tractogram included, are
   deleted once the matrix is made (`--keep` to keep them).
3. The outputs, in `OUT`: `part-<task>.npz` (as above; the sum of the
   subjects' matrices of `schaefer400_17_tian`, the SIFT2 weights between
   two regions times SIFT2's proportionality coefficient so that the
   subjects are comparable, and of `schaefer400_17_tian_count`, the
   streamlines), `matrices/<subject>.npz` (the two matrices of the
   subject), `qc/<subject>.json` (the time of each step, the streamlines
   with both ends assigned, the atlas's Dice, and what the matrix should
   show: the thalamus to the cortex, the striatum to the frontal cortex,
   the hippocampus to the temporal, the interhemispheric share).
4. `python -m hcp_connectivity combine tractography WORK PARTS DATASET`:
   the mean and standard deviation of both matrices
   (`parquet/schaefer400_17_tian[_count]_{mean,sd}.parquet`), the subjects
   (`tractography_subjects.parquet`) and the tables of the engines
   (`tsv/..._tian.tsv.gz`: a parcel and a Tian region; `..._tian_tian`;
   the parcels between them).

## Gradients

`gradients.py`: a principal component analysis of the regions'
connectivity profiles within each hemisphere, the right hemisphere's
rotated onto the left's (see its docstring);
`python -m hcp_connectivity gradients WORK DATASET` makes them again from
the matrices of a dataset.

A new version of the dataset is a new tag there: the engines' URLs and
md5s name it. The TSVs are gzipped without their time, so that a file
made again from the same data is the same.
