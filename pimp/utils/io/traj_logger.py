"""
This class is only necessary as long as this package supports SMAC version <= 0.11.1.
Delete after dropping support for SMAC <= 0.11.1 and use SMAC's TrajLogger instead.

It fixes the interpretation of types in loading the trajectory in the aclib2-format from a file.
Also, it adds the read_traj_alljson_format method, which SMAC supports > 0.11.1 and guarantees backwards-compatibility.
For more details see https://github.com/automl/ParameterImportance/issues/107.
"""
import json
import logging
import typing

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter, CategoricalHyperparameter, Constant
from smac.utils.io.traj_logging import TrajLogger


class TrajLogger(TrajLogger):

    @staticmethod
    def _convert_dict_to_config(config_list: typing.List[str], cs: ConfigurationSpace):
        """Since we save a configurations in a dictionary str->str we have to
        try to figure out the type (int, float, str) of each parameter value
        Parameters
        ----------
        config_list: typing.List[str]
            Configuration as a list of "str='str'"
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object
        """
        config_dict = {}
        for param in config_list:
            k, v = param.split("=")
            v = v.strip("'")
            hp = cs.get_hyperparameter(k)
            if isinstance(hp, FloatHyperparameter):
                v = float(v)
            elif isinstance(hp, IntegerHyperparameter):
                v = int(v)
            elif isinstance(hp, (CategoricalHyperparameter, Constant)):
                # Checking for the correct type requires jumping some hoops
                # First, we gather possible interpretations of our string
                interpretations = [v]
                if v in ["True", "False"]:
                    # Special Case for booleans (assuming we support them)
                    # This is important to avoid false positive warnings triggered by 1 == True or "False" == True
                    interpretations.append(True if v == 'True' else False)
                else:
                    for t in [int, float]:
                        try:
                            interpretations.append(t(v))
                        except ValueError:
                            continue

                # Second, check if it's in the choices / the correct type.
                legal = {l for l in interpretations if hp.is_legal(l)}

                # Third, issue warnings if the interpretation is ambigious
                if len(legal) != 1:
                    logging.getLogger("pimp.trajlogger").warning(
                        "Ambigous or no interpretation of value {} for hp {} found ({} possible interpretations). "
                        "Passing string, but this will likely result in an error".format(v, hp.name, len(legal)))
                else:
                    v = legal.pop()

            config_dict[k] = v

        config = Configuration(configuration_space=cs, values=config_dict)
        config.origin = "External Trajectory"

        return config


    @staticmethod
    def read_traj_alljson_format(fn: str, cs: ConfigurationSpace):
        """Reads trajectory from file
        Parameters
        ----------
        fn: str
            Filename with saved runhistory in self._add_in_alljson_format format
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object
        Returns
        -------
        trajectory: list
            Each entry in the list is a dictionary of the form
            {
            "cpu_time": float,
            "wallclock_time": float,
            "evaluations": int
            "cost": float,
            "incumbent": Configuration
            }
        """

        trajectory = []
        with open(fn) as fp:
            for line in fp:
                entry = json.loads(line)
                entry["incumbent"] = Configuration(cs, entry["incumbent"])
                trajectory.append(entry)

        return trajectory