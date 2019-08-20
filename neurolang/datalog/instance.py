from collections.abc import Set, Mapping


class SetFactCollection(Set):
    def __init__(self, elements):
        self.elements = elements

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, value):
        return value in self.elements

    def __len__(self):
        return len(self.elements)

    def as_mapping(self):
        return MappingFactCollection(self.elements)


class MappingFactCollection(Mapping):
    def __init__(self, elements):
        self.elements = elements

    def __getitem__(self, k):
        return self.elements.get(k)

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def as_set(self):
        return SetFactCollection(self.elements)
