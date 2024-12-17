import random
from pddlgym import structs

class ObjectGenerator:

    def __init__(self, num_objects, types):

        self.num_objects = num_objects
        self.type_list = list(types.values())

    def generate_name(self, idx):

        return "obj%u-o" % (idx)

    def generate_objects(self):

        objects = []

        for idx in range(self.num_objects):

            obj_name = self.generate_name(idx)
            obj_type = random.choice(self.type_list)

            objects.append(structs.TypedEntity(obj_name, obj_type))

        return objects
