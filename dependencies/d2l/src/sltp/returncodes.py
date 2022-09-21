"""
    The possible exit codes of the pipeline components.
"""
from enum import Enum, unique


@unique
class ExitCode(Enum):
    Success = 0

    IterativeMaxsatApproachSuccessful = 2

    MaxsatModelUnsat = 10
    NoAbstractionUnderComplexityBound = 11

    AbstractionFailsOnTestInstances = 20

    # The policy encounters a loop:
    AbstractPolicyNonTerminatingOnTestInstances = 26

    # The policy dictates some abstract action which is not sound wrt some state found in the test instances:
    AbstractPolicyNotSoundOnTestInstances = 27

    # The policy is not total on the set of reached states:
    AbstractPolicyNotCompleteOnTestInstances = 28

    SeparationPolicyNotComplete = 30
    SeparationPolicyCannotDistinguishGoal = 31
    SeparationPolicyAdvisesDeadState = 32

    FeatureGenerationUnknownError = 98
    CNFGenerationUnknownError = 99

    OutOfMemory = 100
    OutOfTime = 101
