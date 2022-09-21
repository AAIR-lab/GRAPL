from generalized_learning.simulator.blocksworld import BlocksworldSimulator
from generalized_learning.simulator.generic import GenericSimulator
from generalized_learning.simulator.gripper import GripperSimulator
from generalized_learning.simulator.hallway import HallwaySimulator
from generalized_learning.simulator.miconic import MiconicSimulator
from generalized_learning.simulator.visitall import VisitallSimulator


def get_simulator(simulator_type):

    if "generic" == simulator_type:

        return GenericSimulator()
    elif "gripper" == simulator_type:

        return GripperSimulator()
    elif "blocksworld" == simulator_type:

        return BlocksworldSimulator()
    elif "miconic" == simulator_type:

        return MiconicSimulator()
    elif "visitall" == simulator_type:

        return VisitallSimulator()
    elif "hallway" == simulator_type:

        return HallwaySimulator()
    elif "rddl" == simulator_type:

        return None
    else:

        raise Exception("Unknown simulator type.")
