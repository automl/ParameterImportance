from pimp.epm.epar_x_rfwi import EPARrfi
from pimp.epm.unlogged_rfwi import Unloggedrfwi as Urfi

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class UnloggedEPARXrfi(EPARrfi, Urfi):

    def __init__(self, configspace, types, bounds, seed,
                 cutoff=0,
                 threshold=0, **kwargs):
        """
        TODO
        """
        Urfi.__init__(self, configspace=configspace, types=types, bounds=bounds, seed=seed, **kwargs)
        EPARrfi.__init__(self, configspace=configspace, types=types, bounds=bounds, seed=seed, cutoff=cutoff, threshold=threshold, **kwargs)

    def predict(self, X):
        """
        Method to override the predict method of RandomForestWithInstances.
        Thus it can be used in the marginalized over instances method of the RFWI class
        """
        return self._predict_EPAR(self._predict(X))
