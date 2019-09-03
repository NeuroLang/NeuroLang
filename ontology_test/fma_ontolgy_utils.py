from owlready2 import  get_ontology

class FMAOntology():

    def __init__(self, path):
        self._ontology = get_ontology(path)
        self.loaded = False

    def load_ontology(self):
        if not self.loaded:
            self._ontology.load()

    def get_nodes(self, main_label):
        if self.loaded:
            return self.process_nodes(main_label)

    def get_subclasses(self, main_label):
        if self.loaded:
            return self.process_subclasses(main_label, apply=self.parse_isSub)

    def get_synonyms(self, main_label):
        if self.loaded:
            return self.process_synonyms(main_label, apply=self.parse_isSyn)

    def process_nodes(self, main_label):
        labels=[]
        for main in self._ontology.search(label = main_label):
            subs = list(main.subclasses())
            while subs:
                subc = subs.pop(0)
                labels.append(subc.label[0])
                subs = subs + list(subc.subclasses())
        return labels

    def process_subclasses(self, main_label, apply=None):
        subclasses = dict()
        for main in self._ontology.search(label = main_label):
            subs = [main]
            while subs:
                subc = subs.pop(0)
                temp = list(subc.subclasses())
                if len(temp) > 0:
                    subclasses[subc.label[0]] = [e.label[0] for e in temp]
                    subs = subs + temp
        if apply is not None:
            subclasses = apply(subclasses)
        return subclasses

    def parse_isSub(self, subc):
        subclass = []
        for k, v in subc.items():
            for elem in v:
                subclass.append((k, elem))
        return subclass

    def process_synonyms(self, main_label, apply=None):
        synonyms = dict()
        for main in self._ontology.search(label = main_label):
            subs = [main]
            while subs:
                subc = subs.pop(0)
                temp = list(subc.subclasses())
                syn = subc.synonym
                if len(syn) > 0:
                    synonyms[subc.label[0]] = syn
                subs = subs + temp
        if apply is not None:
            synonyms = apply(synonyms)
        return synonyms

    def parse_isSyn(self, syn):
        syns = []
        for k, v in syn.items():
            for elem in v:
                syns.append((k, elem))
                syns.append((elem, k))
        return syns
