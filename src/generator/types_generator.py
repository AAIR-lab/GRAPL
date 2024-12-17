
from pddlgym import structs

class TypesGenerator:

    def __init__(self, num_types):

        assert num_types == 1
        self.num_types = num_types

    def generate_name(self, idx):

        return "type%u-t" % (idx)

    def generate_types(self):

        assert self.num_types == 1
        type_hierarchy = {}
        type_to_parent_types = {}
        types = {}

        for idx in range(self.num_types):

            type_name = self.generate_name(idx)
            types[type_name] = structs.Type(type_name)
            type_to_parent_types[type_name] = set([types[type_name]])

        return types, type_hierarchy, type_to_parent_types