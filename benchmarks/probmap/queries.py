"""The query shapes traced (trace.py) and recomputed by counting
(counting.py), on the coordinate meta-analysis engines."""

QUERIES = {
    # benchmarks/memory/queries/pain_map.dl (examples.yaml)
    "pain_map": """
activation(i, j, k, s) :- activated(i, j, k, s), selected_study(s)
term_association(s, t) :- mentions(s, t), selected_study(s)
ans(i, j, k, probability) :- PROB[activation(i, j, k, s) // term_association(s, 'pain')] = probability
""",
    # benchmarks/memory/queries/decode_mask.dl: a deterministic mask
    "decode_mask": """
image_active(i, j, k) :- brain_mask(i, j, k), (k > 45)
image_hit(s) :- selected_voxel(i, j, k), image_active(i, j, k), activated(i, j, k, s)
p_image_given_term(t, PROB) :- (image_hit(s) & selected_study(s)) // (term_in_study(s, t) & selected_study(s))
p_image(PROB) :- image_hit(s) & selected_study(s)
term_studies(t, count(s)) :- term_in_study(s, t)
ans(term, lift, studies) :- p_image_given_term(term, a), p_image(b), term_studies(term, studies), (studies >= 30), (lift == a / b)
""",
    # engines.yaml image_decoding_query: each voxel of the image a
    # probabilistic fact, active with its probability
    "decode_pfact": """
image_probability(i, j, k, p) :- brain_mask(i, j, k), (k > 45), (p == (k - 45) / 50)
image_active(i, j, k) :: p :- image_probability(i, j, k, p)
image_hit(s) :- selected_voxel(i, j, k), image_active(i, j, k), activated(i, j, k, s)
p_image_given_term(t, PROB) :- (image_hit(s) & selected_study(s)) // (term_in_study(s, t) & selected_study(s))
p_image(PROB) :- image_hit(s) & selected_study(s)
term_studies(t, count(s)) :- term_in_study(s, t)
ans(term, lift, studies) :- p_image_given_term(term, a), p_image(b), term_studies(term, studies), (studies >= 30), (lift == a / b)
""",
    # marginal of a voxel: p_a of engines.yaml
    "p_a": """
ans(i, j, k, p) :- PROB[activated(i, j, k, s) & selected_study(s)] = p
""",
    # term co-occurrence (examples.yaml)
    "fear_cooc": """
term_association(s, t) :- mentions(s, t), selected_study(s)
ans(term, probability) :- PROB[term_association(s, term) // term_association(s, 'fear')] = probability
""",
    # p_term of engines.yaml
    "p_term": """
ans(t, p) :- PROB[term_in_study(s, t) & selected_study(s)] = p
""",
    # p_any_voxel_given_t of engines.yaml, for one term
    "any_voxel_pain": """
ans(PROB) :- (activated(i, j, k, s) & selected_voxel(i, j, k) & selected_study(s)) // (term_in_study(s, 'pain') & selected_study(s))
""",
    # region lift (examples.yaml)
    "region_lift": """
active(l, s) :- destrieux_activated(l, s), selected_study(s)
term_association(s, t) :- mentions(s, t), selected_study(s)
p_active_given(l, p) :- PROB[active(l, s) // term_association(s, 'default mode')] = p
p_active(l, p0) :- PROB[active(l, s)] = p0
ans(region, lift) :- p_active_given(region, p), p_active(region, p0), (lift == p / p0)
""",
    # region pairs (examples.yaml): a self-join through the choice
    "co_active": """
active(l, s) :- tian_activated(l, s), selected_study(s)
co_active(l1, l2, s) :- active(l1, s), active(l2, s)
region_probability(l, p) :- PROB[active(l, s)] = p
pair_probability(l1, l2, p) :- PROB[co_active(l1, l2, s)] = p
ans(region_1, region_2, lift) :- pair_probability(region_1, region_2, p12), region_probability(region_1, p1), region_probability(region_2, p2), (region_1 < region_2), (lift == p12 / (p1 * p2))
""",
    # noisy-or: a study hits the image when any of its voxels is active
    # (no selected_voxel): an existential over a probabilistic fact
    "noisy_or": """
image_probability(i, j, k, p) :- brain_mask(i, j, k), (k > 60), (p == (k - 60) / 30)
image_active(i, j, k) :: p :- image_probability(i, j, k, p)
any_hit(s) :- image_active(i, j, k), activated(i, j, k, s)
ans(s, PROB) :- any_hit(s)
""",
}
