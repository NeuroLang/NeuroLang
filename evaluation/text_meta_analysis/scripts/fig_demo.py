import numpy as np, nibabel as nib, nilearn.plotting as npl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("/tmp/exp/demo_maps.npz"); aff = d["affine"]
items = [("D1 free text: 'cortical response to noxious heat applied to the skin'\n(516 studies, size-matched to the term)", "D1_text_pain"), ("Neurosynth term 'pain' (516 studies)", "D1_term_pain"),
         ("D2 free text 'holding information in mind over a delay'\nminus the studies tagged 'working memory' (1,869 studies)", "D2_untagged"), ("Neurosynth term 'working memory' (1,091 studies)", "D2_term_wm"),
         ("D3 free text 'acquisition and extinction of conditioned shock\nexpectancy' (studies near the text)", "D3_text_fear"), ("D3 same, restricted to studies with an amygdala peak (133)", "D3_text_fear_amygdala")]
fig, axes = plt.subplots(6, 1, figsize=(7.5, 16))
for ax, (t, k) in zip(axes, items):
    m = d[k]; thr = np.quantile(m[m > 0], 0.95)
    npl.plot_glass_brain(nib.Nifti1Image(m.astype("float32"), aff), threshold=thr, axes=ax, display_mode="ortho", colorbar=True, cmap="hot"); ax.set_title(t, fontsize=9)
fig.savefig("/tmp/exp/demo_maps.png", dpi=100, bbox_inches="tight"); print("ok")
for k in d.files:
    if k != "affine": nib.save(nib.Nifti1Image(d[k].astype("float32"), aff), f"/tmp/exp/map_{k}.nii.gz")
