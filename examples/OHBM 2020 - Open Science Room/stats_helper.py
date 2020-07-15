import pandas as pd
import nibabel as nib
import numpy as np
import pickle

from rdflib import OWL, RDFS
from nilearn import datasets, plotting
from matplotlib import pyplot as plt
from scipy import special
from scipy.stats import norm

def p_to_z(p, sign):
    p = p / 2  # convert to two-tailed
    # prevent underflow
    p[p < 1e-240] = 1e-240
    # Convert to z and assign tail
    z = np.abs(norm.ppf(p)) * sign
    # Set very large z's to max precision
    z[np.isinf(z)] = norm.ppf(1e-240) * -1
    return z

def one_way(data, n):
    """ One-way chi-square test of independence.
    Takes a 1D array as input and compares activation at each voxel to
    proportion expected under a uniform distribution throughout the array. Note
    that if you're testing activation with this, make sure that only valid
    voxels (e.g., in-mask gray matter voxels) are included in the array, or
    results won't make any sense!
    """
    term = data.astype('float64')
    no_term = n - term
    t_exp = np.mean(term, 0)
    t_exp = np.array([t_exp, ] * data.shape[0])
    nt_exp = n - t_exp
    t_mss = (term - t_exp) ** 2 / t_exp
    nt_mss = (no_term - nt_exp) ** 2 / nt_exp
    chi2 = t_mss + nt_mss
    return special.chdtrc(1, chi2)

def fdr(p, q=.05):
    """ Determine FDR threshold given a p value array and desired false
    discovery rate q. """
    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype='float') * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1

def parse_neurosynth_result(prob_terms_voxels):
    file = open("./data/xyz_from_neurosynth.pkl",'rb')
    ret = pickle.load(file)
    file.close()
    
    regions = []
    for index, row in prob_terms_voxels.iterrows():
        prob = row['prob']
        v_name = row['variable']
        region = ret[v_name]
        regions.append((region, prob))

    prob_img_ns = nib.spatialimages.SpatialImage(
        np.zeros(regions[0][0].image_dim, dtype=float),
        affine=regions[0][0].affine
    )

    for r, p in regions:
        prob_img_ns.dataobj[tuple(r.voxels.T)] = p
    
    return prob_img_ns

def parse_neurolang_result(result, prob_terms):
    result_data = result.value.to_numpy()
    termProb = prob_terms.proba.values[0]


    prob_img = nib.spatialimages.SpatialImage(
        np.zeros(result_data[0][4].image_dim, dtype=float),
        affine=result_data[0][4].affine
    )

    for p in result_data:
        prob_img.dataobj[tuple(p[4].voxels.T)] = p[0]/termProb
        
    return prob_img

def compute_p_values(prob_img, q=1e-25, n=10000):
    prob = prob_img.get_data()
    prob_mask = prob > 0
    prob_mask_values = prob[prob_mask]

    res = one_way(np.round(prob_mask_values * n), n)
    p_values_corrected = fdr(res, q=q)

    p_value_image = np.zeros_like(prob)
    p_value_image[prob_mask] = -np.log10(res)

    p_value_image = nib.spatialimages.SpatialImage(p_value_image, affine=prob_img.affine)
    
    return res, p_values_corrected, p_value_image
