# -*- coding: utf-8 -*-
r"""
Bayes Factor Decoding of the Right Fusiform Gyrus in SQUALL Controlled English
===============================================================================

Performs reverse-inference decoding of the right fusiform gyrus from the
`Julich-Brain v2.9 <https://doi.org/10.1126/science.abb4588>`_ atlas using
Bayes Factors, expressed entirely in
`SQUALL controlled natural language <https://doi.org/10.18653/v1/2020.acl-main.235>`_.

For each cognitive term in the Neurosynth database the Bayes Factor quantifies
the evidence that the right fusiform gyrus is specifically associated with that
term:

.. math::

   \mathrm{BF}(r, t) =
   \frac{P(T{=}t \mid R{=}r)}{P(T{=}t \mid R{\neq}r)}
   =
   \frac{P(R,T)/P(R)}{(P(T) - P(R,T))/(1 - P(R))}

Following Jeffreys (1961) a threshold :math:`\mathrm{BF} > \sqrt{10} \approx 3.16`
indicates "substantial" evidence of association.

The three probability distributions — joint :math:`P(R,T)`, marginal
:math:`P(R)`, and marginal :math:`P(T)` — are each expressed as a single SQUALL
sentence.  Region selection is placed in the ``obtain`` clause so that
NeuroLang's magic-sets optimisation pushes the filter backwards through all three
rules, limiting computation to the target region.
Compare with :ref:`sphx_glr_auto_examples_plot_squall_cbma_spatial_prior.py`
which demonstrates the same SQUALL machinery for a spatial-prior computation.

The Julich-Brain atlas represents the fusiform gyrus as four cytoarchitectonic
areas: ``Area FG1 (FusG) right``, ``Area FG2 (FusG) right``,
``Area FG3 (FusG) right``, and ``Area FG4 (FusG) right``.  These are unioned
into a single ``'right fusiform gyrus'`` label for the decoding analysis.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Region_term_cooccurrence with a probability of
        every Selected_study that ~activates ?r and ~mentions ?t
        for each ?r and for each ?t.

    define as Region_prevalence with a probability of
        every Selected_study that ~activates ?r
        for each ?r.

    define as Term_prevalence with a probability of
        every Selected_study that ~mentions ?t
        for each ?t.

    obtain every Region_term_cooccurrence (?r; ?t; ?p_rt)
        and every Region_prevalence (?r; ?p_r)
        and every Term_prevalence (?t; ?p_t)
        where ?r is 'right fusiform gyrus'.

The three ``define`` sentences use ``Selected_study`` as the grammatical
subject (existentially quantified away) so no study variable appears in the
rule heads.  ``~activates`` and ``~mentions`` are transitive verbs whose
subjects are studies — matching the ``study_activates(study_id, region)``
and ``study_mentions(study_id, term)`` extensional relations.
The Bayes Factor is computed in Python from the three retrieved probability
columns.
"""

# %%
import warnings
from pathlib import Path

import nibabel
import nibabel.affines
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd
import siibra

from neurolang.frontend import NeurolangPDL
from neurolang.frontend.neurosynth_utils import (
    get_ns_mni_peaks_reported,
    get_ns_term_study_associations,
)

warnings.filterwarnings("ignore")

# %%
# Constants
# ---------

data_dir = Path.home() / "neurolang_data"
JULICH_VERSION = "2.9"

# The Julich-Brain atlas represents the fusiform gyrus as four cytoarchitectonic
# areas; we union them under a single label for the analysis.
FUSIFORM_AREAS = [
    "Area FG1 (FusG) right",
    "Area FG2 (FusG) right",
    "Area FG3 (FusG) right",
    "Area FG4 (FusG) right",
]
TARGET_LABEL = "right fusiform gyrus"

BF_THRESHOLD = np.sqrt(10)   # Jeffreys "substantial" evidence ≈ 3.16
TOP_N = 20                   # terms to display in bar chart
