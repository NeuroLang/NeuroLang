"""
Queries for the general notebook
================================
"""

# TODO: Pending to translate all of them


def Q_inferior_temporal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Lateral_fissure_during_y_dominant)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_lateral_than(y, x))
        ),
    )

    return query2.do()


def Q_precentral(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            (
                nl.symbols.isin(
                    x,
                    nl.symbols.Anterior_horizontal_ramus_LF_posterior_dominant,
                )
                | nl.symbols.isin(
                    x, nl.symbols.Anterior_vertical_ramus_LF_during_y_dominant
                )
            )
            & ~nl.symbols.isin(x, nl.symbols.Central_sulcus_posterior_dominant)
            & nl.symbols.anatomical_anterior_of(x, nl.symbols.L_Lat_Fis_post)
            & ~nl.symbols.anatomical_anterior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & nl.symbols.isin(x, nl.symbols.Lateral_fissure_medial_dominant)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()
    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_superior_than(y, x))
        ),
    )

    return query2.do()


def Q_superior_frontal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Central_sulcus_anterior_dominant)
            & nl.symbols.isin(x, nl.symbols.Central_sulcus_during_x_dominant)
            & ~nl.symbols.isin(x, nl.symbols.Central_sulcus_posterior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_superior_than(y, x))
        ),
    )
    return query2.do()


def Q_occipitotemporal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_S_parieto_occipital
            )
            & nl.symbols.anatomical_anterior_of(
                x, nl.symbols.L_S_parieto_occipital
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & nl.symbols.isin(x, nl.symbols.Central_sulcus_posterior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )
    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_lateral_than(y, x))
        ),
    )

    return query2.do()


def Q_superior_temporal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Lateral_fissure_during_y_dominant)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_lateral_than(y, x))
        ),
    )

    return query2.do()


def Q_subparietal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anterior_of(x, nl.symbols.L_S_parieto_occipital)
            & nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_calcarine)
            & nl.symbols.isin(
                x, nl.symbols.Parieto_occipital_sulcus_during_x_dominant
            )
            & nl.symbols.anatomical_posterior_of(x, nl.symbols.L_S_central)
            & ~nl.symbols.isin(
                x, nl.symbols.Parieto_occipital_sulcus_lateral_dominant
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_jensen(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Central_sulcus_posterior_dominant)
            & nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_calcarine)
            & nl.symbols.isin(
                x, nl.symbols.Parieto_occipital_sulcus_anterior_dominant
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_lateral_than(y, x))
        ),
    )

    return query2.do()


def Q_olfactory(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Lateral_fissure_medial_dominant)
            & nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central)
            & nl.symbols.anatomical_inferior_of(x, nl.symbols.L_S_central)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )
    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_intraparietal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_posterior_of(
                x, nl.symbols.Central_sulcus_posterior_dominant
            )
            & nl.symbols.isin(x, nl.symbols.Lateral_fissure_superior_dominant)
            & nl.symbols.isin(
                x, nl.symbols.Parieto_occipital_sulcus_lateral_dominant
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_superior_than(y, x))
        ),
    )

    return query2.do()


#  'Commented',
def Q_lateral_occipital(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_posterior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.anatomical_inferior_of(x, nl.symbols.L_Lat_Fis_post)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_lateral_than(y, x))
        ),
    )

    return query2.do()


def Q_inferior_frontal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central)
            & (
                nl.symbols.isin(
                    x,
                    nl.symbols.Anterior_horizontal_ramus_LF_superior_dominant,
                )
                | nl.symbols.isin(
                    x,
                    nl.symbols.Anterior_horizontal_ramus_LF_during_z_dominant,
                )
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_Lat_Fis_ant_Vertical
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )
    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


def Q_collateral(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(x, nl.symbols.L_S_calcarine)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )
    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


def Q_postcentral(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Central_sulcus_during_z_dominant)
            & nl.symbols.isin(x, nl.symbols.Central_sulcus_during_y_dominant)
            & nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_calcarine)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.isin(x, nl.symbols.Central_sulcus_anterior_dominant)
            & ~nl.symbols.anatomical_inferior_of(x, nl.symbols.L_Lat_Fis_post)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )
    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_posterior_than(y, x))
        ),
    )

    return query2.do()


def Q_anterior_occipital(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.anatomical_posterior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.isin(x, nl.symbols.Calcarine_sulcus_lateral_dominant)
            & ~nl.symbols.isin(x, nl.symbols.Lateral_fissure_anterior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()
    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_lateral_than(y, x))
        ),
    )

    return query2.do()


def Q_callosomarginal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_pericallosal)
            & nl.symbols.isin(x, nl.symbols.Calcarine_sulcus_medial_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_superior_than(y, x))
        ),
    )

    return query2.do()


def Q_middle_frontal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central)
            & nl.symbols.anatomical_superior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.anatomical_inferior_of(x, nl.symbols.L_S_central)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_superior_than(y, x))
        ),
    )

    return query2.do()


def Q_olfactory(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Central_sulcus_medial_dominant)
            & nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central)
            & nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )
    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_orbital_H_shaped(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Central_sulcus_medial_dominant)
            & nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central)
            & nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


# Commented
def Q_superior_occipital(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_posterior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_calcarine)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_posterior_than(y, x))
        ),
    )

    return query2.do()


def Q_intralingual(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_S_parieto_occipital
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_posterior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


#   Non-Destrieux corresponding sulci


def Q_cingulate_cnl(nl):
    nl.execute_cnl_code(
        """
        if `region(X)`,
           P is_named "L_S_pericallosal",
           X anterior_of P,
           X superior_of P,
           `Callosal_sulcus_during_x_dominant_contains(X)`,
           is not the case that
              `Callosal_sulcus_posterior_dominant_contains(X)`,
           is not the case that `found_sulci(X)`, and
           is not the case that `primary_sulci(X)`
        then `cingulate_candidate(X)`.

        if `region(X)`,
           P is_named "L_S_pericallosal",
           X anterior_of P,
           X superior_of P,
           `Callosal_sulcus_medial_dominant_contains(X)`,
           is not the case that
              `Callosal_sulcus_posterior_dominant_contains(X)`,
           is not the case that `found_sulci(X)`, and
           is not the case that `primary_sulci(X)`
        then `cingulate_candidate(X)`.

        if `cingulate_candidate(X)`,
           `cingulate_candidate(Y)`,
           X is_different_than Y, and
           Y is_more_anterior_than X
        then Y is_another_one_more_anterior_than X.

        if `cingulate_candidate(X)` and
           is not the case that Y is_another_one_more_anterior_than X
        then `ans(X)`.
        """
    )

    # _cnlq2 = """
    #     X where
    #     X is_anterior_of L_S_pericallosal,
    #     X is_superior_of L_S_pericallosal,
    #     X is_in Callosal_sulcus_during_x_dominant
    #          or Callosal_sulcus_medial_dominant, and
    #     is not the case that X is_in Callosal_sulcus_posterior_dominant,
    #                                  found_sulci, or primary_sulci.
    #     """

    x = nl.new_region_symbol("x")
    ans = nl.new_symbol(name="ans")
    return nl.query((x,), ans(x))


def Q_compare_cnl(nl):
    nl.execute_cnl_code(
        """
        if X is_named "L_S_pericallosal"
        then `ans(X)`.
        """
        # """
        # if `region(X)`,
        #    P is_named "L_S_pericallosal", and
        #    X is_anterior_of P
        # then `ans(X)`.
        # """
    )
    x = nl.new_region_symbol("x")
    ans = nl.new_symbol(name="ans")
    return nl.query((x,), ans(x))


def Q_compare_datalog(nl):
    x = nl.new_region_symbol("x")
    p = nl.new_region_symbol("p")
    region = nl.new_symbol(name="region")
    is_named = nl.new_symbol(name="is_named")
    is_anterior_of = nl.new_symbol(name="is_anterior_of")
    return nl.query(
        (x,),
        # region(x) & is_named(p, "L_S_pericallosal") & is_anterior_of(x, p),
        is_named(x, "L_S_pericallosal"),
    )


def Q_cingulate_datalog(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    ans = nl.new_symbol(name="ans")
    nl.query(
        ans(x),
        (
            nl.symbols.anterior_of(x, nl.symbols.L_S_pericallosal)
            & nl.symbols.superior_of(x, nl.symbols.L_S_pericallosal)
            & (
                nl.symbols.Callosal_sulcus_during_x_dominant_contains(x)
                | nl.symbols.Callosal_sulcus_medial_dominant_contains(x)
            )
            & ~nl.symbols.Callosal_sulcus_posterior_dominant_contains(x)
            & ~nl.symbols.found_sulci(x)
            & ~nl.symbols.primary_sulci(x)
        ),
    )

    another_one_more_anterior = nl.new_symbol(name="another_one_more_anterior")
    nl.query(
        another_one_more_anterior(x, y),
        ans(x) & ans(y) & (x != y) & nl.symbols.is_more_anterior_than(y, x),
    )

    return nl.query((x,), ans(x) & ~another_one_more_anterior(x, y),)


def Q_paracingulate(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_anterior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_anterior_than(y, x))
        ),
    )

    return query2.do()


def Q_inferior_occipital(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_S_parieto_occipital
            )
            & nl.symbols.anatomical_posterior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


def Q_anterior_parolfactory(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_anterior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.inferior_of(x, nl.symbols.L_S_pericallosal)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_inferior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


def Q_lunate(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_posterior_of(x, nl.symbols.L_Lat_Fis_post)
            & nl.symbols.isin(x, nl.symbols.Calcarine_sulcus_during_x_dominant)
            & nl.symbols.anatomical_inferior_of(x, nl.symbols.L_S_central)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_posterior_than(y, x))
        ),
    )

    return query2.do()


def Q_cuneal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_posterior_of(x, nl.symbols.L_S_pericallosal)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_during_x_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_frontomarginal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anterior_of(x, nl.symbols.L_S_pericallosal)
            & nl.symbols.superior_of(x, nl.symbols.L_S_pericallosal)
            & (
                nl.symbols.isin(
                    x, nl.symbols.Callosal_sulcus_during_x_dominant
                )
                | nl.symbols.isin(
                    x, nl.symbols.Callosal_sulcus_medial_dominant
                )
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_anterior_than(y, x))
        ),
    )

    return query2.do()


def Q_hippocampal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_posterior_of(
                x, nl.symbols.L_Lat_Fis_ant_Vertical
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_during_y_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


def Q_superior_parietal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_pericallosal)
            & nl.symbols.isin(x, nl.symbols.Central_sulcus_posterior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_anterior_than(y, x))
        ),
    )

    return query2.do()


def Q_rhinal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_anterior_than(y, x))
        ),
    )

    return query2.do()


def Q_temporopolar(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_retrocalcarine(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(
                x, nl.symbols.Parieto_occipital_sulcus_posterior_dominant
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_medial_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_posterior_than(y, x))
        ),
    )

    return query2.do()


def Q_paracentral(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Callosal_sulcus_medial_dominant)
            & nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_pericallosal)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_angular(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            ~nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_S_parieto_occipital
            )
            & nl.symbols.anatomical_posterior_of(x, nl.symbols.L_S_central)
            & nl.symbols.isin(
                x, nl.symbols.Parieto_occipital_sulcus_anterior_dominant
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_lateral_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_inferior_than(y, x))
        ),
    )

    return query2.do()


def Q_inferior_rostral(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central)
            & nl.symbols.anatomical_inferior_of(
                x, nl.symbols.L_Lat_Fis_ant_Horizont
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_inferior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()


def Q_intralimbic(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Callosal_sulcus_superior_dominant)
            & nl.symbols.anatomical_anterior_of(
                x, nl.symbols.L_S_parieto_occipital
            )
            & nl.symbols.isin(x, nl.symbols.Callosal_sulcus_medial_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_medial_than(y, x))
        ),
    )

    return query2.do()
