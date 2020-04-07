from .query_resolution import QueryBuilderBase, RegionMixin, NeuroSynthMixin


class QueryBuilderFirstOrderThroughDatalog(
    RegionMixin, NeuroSynthMixin, QueryBuilderBase
):
    pass
