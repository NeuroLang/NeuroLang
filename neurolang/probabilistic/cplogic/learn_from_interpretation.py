def full_observability_parameter_estimation(
    program_code, interpretations_dict, n_interpretations
):
    grounded = ground_cplogic_program(program_code)
    estimations = []
    for grounding in grounded.expressions:
        if is_probabilistic_fact(grounding.expression):
            estimations.append(
                _infer_pfact_params(
                    grounding, interpretations_dict, n_interpretations
                )
            )
    result = ExtendedRelationalAlgebraSolver({}).walk(
        Aggregation(
            agg_fun=Constant[str]("mean"),
            relation=MultipleUnions(estimations),
            group_columns=[Constant(ColumnStr("__parameter_name__"))],
            agg_column=Constant(ColumnStr("__parameter_estimate__")),
            dst_column=Constant(ColumnStr("__parameter_estimate__")),
        )
    )
    return result


def _infer_pfact_params(
    pfact_grounding, interpretations_dict, n_interpretations
):
    """
    Compute the estimate of the parameters associated with a specific
    probabilistic fact predicate symbol from the facts with that same predicate
    symbol found in interpretations.

    """
    pred_symb = pfact_grounding.expression.consequent.body.functor
    pred_args = pfact_grounding.expression.consequent.body.args
    interpretation_ra_set = Constant[AbstractSet](
        interpretations_dict[pred_symb]
    )
    for arg, col in zip(
        pred_args, [c for c in interpretation_ra_set.value.columns]
    ):
        interpretation_ra_set = RenameColumn(
            interpretation_ra_set,
            Constant[ColumnStr](ColumnStr(col)),
            Constant[ColumnStr](ColumnStr(arg.name)),
        )
    tuple_counts = Aggregation(
        agg_fun=Constant[str]("count"),
        relation=NaturalJoin(pfact_grounding.relation, interpretation_ra_set),
        group_columns=tuple(
            Constant(ColumnStr(arg.name))
            for arg in pfact_grounding.expression.consequent.body.args
        )
        + (
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
        ),
        agg_column=Constant(ColumnStr("__interpretation_id__")),
        dst_column=Constant(ColumnStr("__tuple_counts__")),
    )
    substitution_counts = Aggregation(
        agg_fun=Constant[str]("count"),
        relation=pfact_grounding.relation,
        group_columns=(
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
        ),
        agg_column=None,
        dst_column=Constant(ColumnStr("__substitution_counts__")),
    )
    probabilities = ExtendedProjection(
        NaturalJoin(tuple_counts, substitution_counts),
        tuple(
            [
                ExtendedProjectionListMember(
                    fun_exp=Constant(ColumnStr("__tuple_counts__"))
                    / (
                        Constant(ColumnStr("__substitution_counts__"))
                        * Constant[float](float(n_interpretations))
                    ),
                    dst_column=Constant(ColumnStr("__probability__")),
                )
            ]
        ),
    )
    parameter_estimations = Aggregation(
        agg_fun=Constant[str]("mean"),
        relation=probabilities,
        group_columns=(
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
        ),
        agg_column=Constant(ColumnStr("__probability__")),
        dst_column=Constant(ColumnStr("__parameter_estimate__")),
    )
    solver = ExtendedRelationalAlgebraSolver({})
    return solver.walk(
        RenameColumn(
            parameter_estimations,
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
            Constant(ColumnStr("__parameter_name__")),
        )
    )


def _build_interpretation_ra_sets(grounding, interpretations):
    pred = grounding.expression.consequent.body
    pred_symb = pred.functor
    columns = tuple(arg.name for arg in pred.args)
    itps_at_least_one_tuple = [
        itp.as_map()
        for itp in interpretations
        if pred_symb in itp.as_map().keys()
    ]
    itp_ra_sets = [
        NaturalJoin(
            Constant[AbstractSet](
                ExtendedAlgebraSet(
                    columns, itp[pred_symb].value._container.values
                )
            ),
            Constant[AbstractSet](
                ExtendedAlgebraSet(["__interpretation_id__"], [itp_id])
            ),
        )
        for itp_id, itp in enumerate(itps_at_least_one_tuple)
    ]
    return ExtendedRelationalAlgebraSolver({}).walk(
        NaturalJoin(grounding.relation, MultipleUnions(itp_ra_sets))
    )
