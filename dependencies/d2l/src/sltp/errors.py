

class CriticalPipelineError(Exception):
    """ A critical error which should prevent the computation of further steps in the pipeline """
    pass
