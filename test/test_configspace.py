import os
import numpy as np
import csv
import unittest

from pimp.importance.importance import Importance
from smac.utils.io.input_reader import InputReader
from smac.scenario.scenario import Scenario
from ConfigSpace.read_and_write import json as pcs_json

class TestConfigspace(unittest.TestCase):

    def setUp(self):
        base_dir = "test/configspace_cornercase"

        in_reader = InputReader()
        # Create Scenario (disable output_dir to avoid cluttering)
        scen_fn = os.path.join(base_dir, 'scenario.txt')
        scen_dict = in_reader.read_scenario_file(scen_fn)
        scen_dict['output_dir'] = ""
        # We always prefer the less error-prone json-format if available:
        cs_json = os.path.join(base_dir, 'configspace.json')
        if os.path.exists(cs_json):
            with open(cs_json, 'r') as fh:
                pcs_fn = scen_dict.pop('pcs_fn')
                scen_dict['cs'] = pcs_json.read(fh.read())
        scen = Scenario(scen_dict)
        rh_fn = os.path.join(base_dir, 'runhistory.json')
        #traj_fn = os.path.join(base_dir, 'traj_aclib2.json')

        self.imp = Importance(scenario=scen,
                              runhistory_file=rh_fn,
                              incumbent=scen.cs.sample_configuration(),
                              #traj_file=traj_fn,
                              seed=42)


    def test_mixed_categorical(self):
        """ Having ints and bools as categoricals """

        self.imp.evaluate_scenario(['fanova'], 'test/configspace_cornercase')
