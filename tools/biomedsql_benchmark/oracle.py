"""Oracle: BiomedSQL's gold SQL translated, mechanically, to NeuroLang Datalog.

Not a system under test: it shows that every item is answerable in the engine,
and that the scorers accept a correct answer computed by the engine (an upper
bound on any system), and it supplies the examples of the few-shot condition
(from development templates only).

Covers the SQL of the 40 templates: ``SELECT [DISTINCT] exprs FROM table [WHERE
conjunction of comparisons and LIKEs, with OR groups] [GROUP BY] [ORDER BY]
[LIMIT]``, ``UNION ALL`` of such selects, ``COUNT(*)`` / ``COUNT(DISTINCT c)``,
computed columns (``1-freq``) and literal columns. ORDER BY and LIMIT have no
Datalog form: they are dropped, except that a one-row ORDER BY ... LIMIT 1 keeps
the best row through ``min`` / ``max``. A list cut by its LIMIT is then returned
whole, which the scorers accept (they require the gold rows to be in it).
"""

import re

RELATIONS = {
    "ParkinsonDisease_CompleteGeneData_No23andMe": "pd_gwas",
    "AlzheimerDisease_CombinedGeneData_UUID": "ad_gwas",
    "NeurodegenerativeDiseases_SMR_Genes_Full": "smr",
    "DrugGeneTargets_ComprehensiveAnnotations_updated": "drug",
    "DrugTargets_IndicationsAndTherapeuticUses": "drug_indication",
}


class Unsupported(Exception):
    pass


def split_top(text, keyword):
    """Split *text* on the keyword, outside parentheses and quotes."""
    parts, depth, quote, current, i = [], 0, None, "", 0
    n = len(keyword)
    while i < len(text):
        ch = text[i]
        if quote:
            current += ch
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote, current = ch, current + ch
        elif ch == "(":
            depth, current = depth + 1, current + ch
        elif ch == ")":
            depth, current = depth - 1, current + ch
        elif (
            depth == 0
            and text[i : i + n].upper() == keyword
            and text[i - 1 : i].isspace()
            and text[i + n : i + n + 1].isspace()
        ):
            parts.append(current.strip())
            current, i = "", i + n
            continue
        else:
            current += ch
        i += 1
    parts.append(current.strip())
    return parts


def literal(value):
    if isinstance(value, (int, float)):
        return repr(value)
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    raise Unsupported("both quotes in a literal")


def constant(text):
    m = re.fullmatch(r"(['\"])(.*)\1", text, re.S)
    if m:
        return m.group(2)
    if text in ("True", "False"):
        return text
    number = float(text)
    return int(number) if re.fullmatch(r"-?\d+", text) else number


LIKE = re.compile(
    r"^(?:LOWER\((\w+)\)|(\w+))\s+LIKE\s+(['\"])%(.*)%\3$", re.I | re.S
)
EQ_LOWER = re.compile(r"^LOWER\((\w+)\)\s*=\s*(['\"])(.*)\2$", re.I | re.S)
CMP = re.compile(r"^(\w+)\s*(=|<=|>=|<|>|!=)\s*(.+)$", re.S)


def parse_select(sql):
    m = re.match(
        r"^SELECT (DISTINCT )?(.*?) FROM `[^`]*\.(\w+)`"
        r"(?: ?WHERE ?(.*?))?(?: GROUP BY (\w+))?"
        r"(?: ORDER BY (\w+)( DESC| ASC)?)?(?: LIMIT (\d+))?$",
        sql.strip(),
        re.I | re.S,
    )
    if not m:
        raise Unsupported(sql[:80])
    select = [s.strip() for s in split_commas(m.group(2))]
    conds = []
    for c in split_top(m.group(4), "AND") if m.group(4) else []:
        c = c.strip()
        if c.startswith("(") and c.endswith(")") and " OR " in c.upper():
            conds.append(("or", [atom(a) for a in split_top(c[1:-1], "OR")]))
        else:
            conds.append(atom(c))
    return dict(
        select=select,
        table=m.group(3),
        conds=conds,
        group=m.group(5),
        order=m.group(6),
        desc=(m.group(7) or "").strip().upper() == "DESC",
        limit=int(m.group(8)) if m.group(8) else None,
    )


def split_commas(text):
    out, depth, current = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(current)
            current = ""
        else:
            current += ch
    out.append(current)
    return out


def atom(text):
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    m = LIKE.match(text)
    if m:
        return ("like", m.group(1) or m.group(2), m.group(4), bool(m.group(1)))
    m = EQ_LOWER.match(text)
    if m:
        return ("eq_lower", m.group(1), m.group(3).lower())
    m = CMP.match(text)
    if m:
        return ("cmp", m.group(1), m.group(2), constant(m.group(3).strip()))
    raise Unsupported(text[:60])


def test_text(a, var):
    if a[0] == "like":
        _, col, text, lower = a
        subject = f"lcase({var[col]})" if lower else var[col]
        return f"str_contains({subject}, {literal(text.lower() if lower else text)})"
    if a[0] == "eq_lower":  # lcase(x) == c is not resolved; bind it first
        return f"(l_{var[a[1]]} == lcase({var[a[1]]})), (l_{var[a[1]]} == {literal(a[2])})"
    _, col, op, value = a
    return f"({var[col]} {'==' if op == '=' else op} {literal(value)})"


def block(p, columns, relation, tag):
    """The body, output variables and helper rules of one parsed select."""
    var = {c: f"x_{c.lower()}" for c in columns}
    inline, tests, helpers, extra = {}, [], [], []
    for k, a in enumerate(p["conds"]):
        if a[0] == "or":
            name = f"{tag}_or{k}"
            for alt in a[1]:
                helpers.append(
                    f"{name}({var['UUID']}) :- "
                    f"{relation}({', '.join(var[c] for c in columns)}), {test_text(alt, var)}"
                )
            extra.append(f"{name}({var['UUID']})")
        elif a[0] == "cmp" and a[2] == "=":
            inline[a[1]] = literal(a[3])
        else:
            tests.append(test_text(a, var))
    args = ", ".join(inline.get(c, var[c]) for c in columns)
    body = [f"{relation}({args})"] + extra + tests
    out_vars, computed = [], []
    for e in p["select"]:
        m = re.match(r"^(.*?)\s+AS\s+(\w+)$", e, re.I)
        expr, alias = (m.group(1).strip(), m.group(2)) if m else (e, e)
        if expr == "UUID":
            continue
        if expr in var:
            out_vars.append(var[expr])
            if expr in inline:
                body.append(f"({var[expr]} == {inline[expr]})")
        elif re.fullmatch(r"(['\"]).*\1", expr):
            body.append(f"(x_{alias.lower()} == {literal(constant(expr))})")
            out_vars.append(f"x_{alias.lower()}")
        elif re.fullmatch(r"1\s*-\s*\w+", expr):
            column = re.fullmatch(r"1\s*-\s*(\w+)", expr).group(1)
            body.append(f"(x_{alias.lower()} == 1 - {var[column]})")
            out_vars.append(f"x_{alias.lower()}")
        elif expr.upper().startswith("COUNT("):
            computed.append(expr)
        else:
            raise Unsupported(expr)
    return body, out_vars, computed, var, helpers


def oracle(sql, columns_of):
    """The Datalog program of the gold *sql*; *columns_of* maps a relation to its columns."""
    sql = re.sub(r"\s+", " ", sql).strip()
    parts = split_top(re.sub(r"\sLIMIT \d+$", "", sql), "UNION ALL")
    if len(parts) > 1:
        rules = []
        for k, part in enumerate(parts):
            p = parse_select(part)
            relation = RELATIONS[p["table"]]
            body, out_vars, _, _, helpers = block(
                p, columns_of[relation], relation, f"u{k}"
            )
            rules += helpers + [
                f"u({', '.join(out_vars)}) :- {', '.join(body)}"
            ]
        return "\n".join(
            rules + [f"ans({', '.join(out_vars)}) :- u({', '.join(out_vars)})"]
        )
    p = parse_select(sql)
    relation = RELATIONS[p["table"]]
    body, out_vars, computed, var, helpers = block(
        p, columns_of[relation], relation, "c"
    )
    if computed:  # GROUP BY key, COUNT(*) / COUNT(DISTINCT column)
        distinct = re.fullmatch(r"COUNT\(DISTINCT (\w+)\)", computed[0], re.I)
        counted = var[distinct.group(1)] if distinct else var["UUID"]
        key = var[p["group"]]
        args = ", ".join(dict.fromkeys([key, counted]))
        return "\n".join(
            helpers
            + [
                f"rel({args}) :- {', '.join(body)}",
                f"ans({key}, n) :- AGGREGATE[{key}](rel({args}) @ count({counted})) = n",
            ]
        )
    if p["limit"] == 1 and p["order"]:  # the one best row
        order = var[p["order"]]
        args = ", ".join(dict.fromkeys(out_vars + [order]))
        function = "max" if p["desc"] else "min"
        return "\n".join(
            helpers
            + [
                f"rel({args}) :- {', '.join(body)}",
                f"best(m) :- AGGREGATE[](rel({args}) @ {function}({order})) = m",
                f"ans({', '.join(out_vars)}) :- rel({args}), best(m), ({order} == m)",
            ]
        )
    return "\n".join(
        helpers + [f"ans({', '.join(out_vars)}) :- {', '.join(body)}"]
    )
