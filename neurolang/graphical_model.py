
class GraphicalModel:
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.random_variables = set()
        # map associating a CPD (conditional probability distribution)
        # to each random variable in the network
        self.cpds = dict()
