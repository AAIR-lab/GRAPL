from .zpk import ZPKOperatorLearningModule
from .foldt import FOLDTOperatorLearningModule
from .groundtruth import GroundTruthOperatorLearningModule

def create_operator_learning_module(operator_learning_name, learned_operators, domain_name):
    if operator_learning_name.startswith("groundtruth"):
        env_name = operator_learning_name[len("groundtruth-"):]
        return GroundTruthOperatorLearningModule(env_name, learned_operators)
    if operator_learning_name == "LNDR":
        return ZPKOperatorLearningModule(learned_operators, domain_name)
    if operator_learning_name == "TILDE":
        return FOLDTOperatorLearningModule(learned_operators)
    raise Exception("Unrecognized operator learning module '{}'".format(operator_learning_name))
