

#include "base.h"

#include <boost/algorithm/string.hpp>

namespace sltp {


//! Parse a list of predicate names and return their indexes in the global predicate index
std::vector<predicate_id_t> parse_predicate_names(
        const std::string &line,
        const Sample::PredicateIndex& predicate_index) {
    std::vector<std::string> names;
    boost::split(names, line, boost::is_any_of(" \t"));

    std::vector<predicate_id_t> predicates;
    for(const auto &name : names) {
        if (!name.empty()) {
//            auto pid = (object_id_t) predicate_index.size();
            predicates.push_back(predicate_index.at(name));
        }
    }

    return predicates;
}


Atom parse_atom(const std::string &line,
                const Instance::ObjectIndex &object_index,
                const Sample::PredicateIndex &predicate_index) {
    std::vector<std::string> atom;
    boost::split(atom, line, boost::is_any_of(","));  // Split e.g. "at,bob,shed" into a vector
    if( atom.empty() ) throw std::runtime_error("Wrong atom format: " + line);

    std::vector<object_id_t> objects;
    objects.reserve(atom.size() - 1);
    for( int i = 1; i < atom.size(); ++i ) { // Start iterating at 1 to skip the predicate name
        objects.push_back(object_index.left.at(atom[i]));
    }
    return Atom(predicate_index.at(atom[0]), std::move(objects));
}

std::vector<Atom> parse_atoms(const std::string &line,
                              const Instance::ObjectIndex &object_index,
                              const Sample::PredicateIndex &predicate_index,
                              Instance::AtomIndex &atom_index) {
    std::vector<std::string> atom_list;
    boost::split(atom_list, line, boost::is_any_of("\t"));

    std::vector<Atom> atoms;
    atoms.reserve(atom_list.size());

    for( auto &atom_str : atom_list ) {
        auto aid = (atom_id_t) atoms.size();
        atoms.emplace_back(std::move(parse_atom(atom_str, object_index, predicate_index)));

        // Update the atom index
        atom_index.emplace(atoms.back().data(), aid);
    }

    return atoms;
}

State parse_state(const std::vector<Instance> &instances,
                  unsigned id,
                  const std::string &line,
                  const Sample::PredicateIndex &predicate_index) {
    std::vector<std::string> state_parts;
    boost::split(state_parts, line, boost::is_any_of("\t"));
    assert(!state_parts.empty());  // At least we'll have the ID of the instance

    auto instance_id = (unsigned) std::stoi(state_parts[0]);
    const Instance &instance = instances.at(instance_id);
    const auto &object_index = instance.object_index();
    const auto &atom_index = instance.atom_index();

    std::vector<atom_id_t> atom_ids;
    atom_ids.reserve(state_parts.size());
    for( unsigned i = 1; i < state_parts.size(); ++i ) {
        const auto &atom_str = state_parts[i];
        auto atom = parse_atom(atom_str, object_index, predicate_index);
        atom_ids.push_back(atom_index.at(atom.data()));
    }

    return State(instance_id, id, std::move(atom_ids));
}

std::vector<Object> parse_objects(const std::string &line, Instance::ObjectIndex &object_index) {
    std::vector<std::string> names;
    boost::split(names, line, boost::is_any_of(" \t"));

    std::vector<Object> objects;
    for( auto &name : names ) {
        auto oid = (object_id_t) object_index.size();
        objects.emplace_back(oid, name);
        object_index.insert({name, oid});
    }

    return objects;
}

std::vector<Predicate> parse_predicates(const std::string &line, Sample::PredicateIndex& predicate_index) {
    std::vector<std::string> names;
    boost::split(names, line, boost::is_any_of(" \t"));

    std::vector<Predicate> predicates;
    for( auto &str : names ) {
        std::vector<std::string> info;
        boost::split(info, str, boost::is_any_of("/"));  // Expected format: "clear/1"
        if( info.size() != 2 ) throw std::runtime_error("Wrong predicate format: " + str);

        auto pid = (object_id_t) predicate_index.size();
        predicates.emplace_back(pid, info[0], (unsigned) std::stoi(info[1]));
        predicate_index.insert(std::make_pair(info[0], pid));
    }

    return predicates;
}


Sample Sample::read(std::istream &is) {
    //ObjectIndex object_index;
    PredicateIndex predicate_index;
    //AtomIndex atom_index;

    // First line: list of all predicate names (shared by all instances)
    // (Note: this will include functions as well, if used in the problem representation)
    std::string predicate_line;
    std::getline(is, predicate_line);
    std::vector<Predicate> predicates(std::move(parse_predicates(predicate_line, predicate_index)));

    // Next: List of all predicates and functions mentioned in the goal
    std::string line;
    std::getline(is,line);
    std::vector<predicate_id_t> goal_predicates = parse_predicate_names(line, predicate_index);

    // Next: number of instances in file
    std::string num_instances_line;
    std::getline(is, num_instances_line);
    int num_instances = atoi(num_instances_line.c_str());

    // The next block contains two lines per each instance:
    // - one line with all object names in that instance
    // - one line with all possible atoms in that instance, including statics and types
    std::vector<Instance> instances;
    for( int i = 0; i < num_instances; ++i ) {
        Instance::ObjectIndex object_index;
        Instance::AtomIndex atom_index;

        // Read all possible object names in this instance
        std::string object_line;
        std::getline(is, object_line);
        std::vector<Object> objects = parse_objects(object_line, object_index);
        //std::cout << "instance: #objects=" << objects.size() << std::endl;

        // Read all possible atoms in instance
        std::string atom_line;
        std::getline(is, atom_line);
        std::vector<Atom> atoms = parse_atoms(atom_line, object_index, predicate_index, atom_index);
        //std::cout << "instance: #atoms=" << atoms.size() << std::endl;

        // Create and store the instance
        instances.emplace_back(i, std::move(objects), std::move(atoms), std::move(object_index), std::move(atom_index));
    }

    // Finally, we have one line per state. Each line consists of one integer that denotes to which instance
    // the state belongs to, plus all the atoms in that state, including static and type-based atoms
    std::vector<State> states;
    std::string state_line;
    for( unsigned id = 0; std::getline(is, state_line); ++id ) {
        states.emplace_back(std::move(parse_state(instances, id, state_line, predicate_index)));
    }

    // create and return sample
    return Sample(std::move(predicates),
                  std::move(instances),
                  std::move(states),
                  std::move(goal_predicates),
                  std::move(predicate_index));
}


}
