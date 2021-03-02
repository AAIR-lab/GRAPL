
import random

from benchmarks.generator import Generator
from util import constants
from util import file


class DeliveryDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_crates", "max_crates", "locations"]) \
        .union(Generator.REQUIRED_KEYS)

    MIN_CRATES = 1

    _DOMAIN_NAME = "delivery"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(DeliveryDomainGenerator, self).__init__(parent, parent_dir,
                                                      global_dict,
                                                      user_phase_dict,
                                                      failfast)

    def generate_domain(self):

        file_handle = open("%s/%s.%s" % (
            self._base_dir,
            DeliveryDomainGenerator._DOMAIN_NAME,
            constants.DOMAIN_FILE_EXT), "w")

        predicates = [" (crate ?x)", " (truck ?x)", " (in_truck ?x)"]
        actions = []

        locations = self.get_value("locations")
        file_handle.write("""(define (domain %s)

    (:requirements :equality)

""" % (DeliveryDomainGenerator._DOMAIN_NAME))

        for location in range(locations):

            predicates.append(" (atl{} ?x)".format(location))

            actions.append("""    (:action movetol%u
        :parameters (?t)
        :precondition (and (truck ?t) (not (atl%u ?t)))
        :effect (and (atl%u ?t)%s))
    
""" % (location,
                location,
                location,
                "".join([" (not (atl%u ?t))" % (i)
                         for i in range(locations) if location != i])))

            actions.append("""    (:action loadatl%u
        :parameters (?c ?t)
        :precondition (and (crate ?c) (atl%u ?c) (truck ?t) (atl%u ?t))
        :effect (and (in_truck ?c) (not (atl%u ?c))))
    
""" % (location, location, location, location))

            actions.append("""    (:action unloadatl%u
        :parameters (?c ?t)
        :precondition (and (crate ?c) (in_truck ?c) (truck ?t) (atl%u ?t))
        :effect (and (atl%u ?c) (not (in_truck ?c))))

""" % (location, location, location))

        file_handle.write("""    (:predicates%s)

""" % ("".join(predicates)))
        file_handle.write("%s" % ("".join(actions)))
        file_handle.write(")")
        file_handle.close()

    def generate_problem(self, problem_range):

        locations = self.get_value("locations")
        last_location = locations - 1
        min_crates = self.get_value("min_crates")
        max_crates = self.get_value("max_crates")

        assert min_crates >= DeliveryDomainGenerator.MIN_CRATES

        for problem_no in problem_range:

            crates = random.randint(min_crates, max_crates)
            assert crates > 0

            # Generate a list of start locations and destinations.
            source_list = []
            destination_list = []
            for _ in range(crates):

                source_list.append(random.randint(0, last_location))
                destination_list.append(random.randint(0, last_location))

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            properties = {

                "locations": locations,
                "min_crates": min_crates,
                "max_crates": max_crates,
                "crates": crates,

                "bin_params": ["crates"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            file_handle.write("""(define (problem delivery-%uc-%ul)
    
""" % (crates, locations))

            file_handle.write("""    (:domain %s)

""" % (DeliveryDomainGenerator._DOMAIN_NAME))

            file_handle.write("""    (:objects truck0%s)

""" % ("".join([" crate%u" % (i) for i in range(crates)])))

            file_handle.write("""    (:init (truck truck0) (atl%u truck0)%s)

""" % (random.randint(0, last_location),
                "".join([" (crate crate%u) (atl%u crate%u)" % (
                    i, source_list[i], i)
                    for i in range(crates)])))

            file_handle.write("""    (:goal (and%s))

""" % ("".join([" (atl%u crate%u)" % (destination_list[i], i)
                for i in range(crates)])))

            file_handle.write(")")
            file_handle.close()

        # Just return an empty list.
        return []
