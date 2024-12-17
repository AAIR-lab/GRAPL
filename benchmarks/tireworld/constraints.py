
def has_one_vehicle(state, _action):

    vehicle_count = 0
    for literal in state:
        if literal.predicate.name == "vehicle-at":
            vehicle_count += 1

            if vehicle_count > 1:
                return False

    return vehicle_count == 1