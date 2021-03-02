'''
Created on Feb 7, 2020

@author: rkaria
'''

from util.phase import PhaseManager

SAMPLE_YAML_CONFIG = """
globals:
    name: "test"
    ignore: True

phases:
    - phase_1:
        name: "phase_1"

        sub_phases:
            - name: "sub_phase_1"
              data: [1, 2, 3, "data", False]

            - name: "sub_phase_2"
              data: None

    - phase_2:
        name: "phase_2"
"""


class ExamplePhaseManager(PhaseManager):

    def __init__(self, yaml_config):

        super(ExamplePhaseManager, self).__init__(yaml_config)

    def run(self):

        print(self._global_dict)
        print(self._phase_dict)


if __name__ == "__main__":

    print("Hi")
    example_phase_manager = ExamplePhaseManager(SAMPLE_YAML_CONFIG)
    example_phase_manager.run()
