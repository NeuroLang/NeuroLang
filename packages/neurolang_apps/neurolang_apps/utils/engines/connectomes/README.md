# Structural connectivity of the cortex (HCP)

`hcp_schaefer100_17.tsv.gz` and `hcp_destrieux.tsv.gz` are the relations
`schaefer_structural_connectivity` and `destrieux_structural_connectivity`
of the engines with the MNI atlases (`mni_atlases` in `../engines.yaml`):

| column | meaning |
| --- | --- |
| `region1`, `region2` | two distinct parcels, named as the engines' `schaefer` and `destrieux` labels |
| `streamlines` | the mean, over the subjects, of the number of streamlines of tractography joining them |

Each pair is there both ways round; the pairs no streamline joins are left
out. Each subject has 2.8 to 3.6 million streamlines.

## How they were made

They are the high-resolution structural connectomes of 1,064 subjects of
the Human Connectome Project (Young Adult, 3T diffusion MRI) by Mansour et
al. (2021): MRtrix3 tractography, the streamlines' endpoints mapped to the
59,412 cortical vertices of the fs_LR 32k surface and smoothed on it, in
the authors' public bucket. `tools/hcp_structural_connectivity` sums each
subject's matrix between the vertices of two parcels and averages the
subjects:

- Schaefer 2018, 100 parcels, 17 networks: its fs_LR 32k labels (CBIG);
- Destrieux 2009: nilearn's labels of fsaverage5, brought to fs_LR 32k by
  the nearest vertex on the sphere through the HCP's registration of fs_LR
  to fsaverage.

The connectomes have no subcortex: there is no Tian connectivity. Checked
against the authors' own parcellation with Glasser's atlas, subject by
subject (`validation.tsv` of the tools), the streamlines between distinct
parcels correlate at r = 0.999.

## Acknowledgements

Using these relations, cite Mansour et al. (2021) and the atlas, and
acknowledge the Human Connectome Project as its data use terms ask
(`.credits` in the REPL prints them):

- Mansour L S, Tian Y, Yeo BTT, Cropley V, Zalesky A. High-resolution
  connectomic fingerprints: mapping neural identity and behavior.
  *NeuroImage* 229:117695 (2021). doi:10.1016/j.neuroimage.2020.117695
- Van Essen DC, Smith SM, Barch DM, Behrens TEJ, Yacoub E, Ugurbil K. The
  WU-Minn Human Connectome Project: an overview. *NeuroImage* 80:62-79
  (2013). doi:10.1016/j.neuroimage.2013.05.041
- Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ, Eickhoff
  SB, Yeo BTT. Local-global parcellation of the human cerebral cortex from
  intrinsic functional connectivity MRI. *Cerebral Cortex* 28:3095-3114
  (2018). doi:10.1093/cercor/bhx179
- Destrieux C, Fischl B, Dale A, Halgren E. Automatic parcellation of human
  cortical gyri and sulci using standard anatomical nomenclature.
  *NeuroImage* 53:1-15 (2010). doi:10.1016/j.neuroimage.2010.06.010
- Van Essen DC, Glasser MF, Dierker DL, Harwell J, Coalson T. Parcellations
  and hemispheric asymmetries of human cerebral cortex analyzed on
  surface-based atlases. *Cerebral Cortex* 22:2241-2262 (2012).
  doi:10.1093/cercor/bhr291 (the fs_LR to fsaverage registration)

> Data were provided [in part] by the Human Connectome Project, WU-Minn
> Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil;
> 1U54MH091657) funded by the 16 NIH Institutes and Centers that support
> the NIH Blueprint for Neuroscience Research; and by the McDonnell Center
> for Systems Neuroscience at Washington University.

The HCP's Open Access Data Use Terms apply to the data these relations are
derived from (https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
