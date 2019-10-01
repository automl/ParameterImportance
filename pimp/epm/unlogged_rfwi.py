import numpy as np
from scipy import stats

from pimp.epm.base_epm import RandomForestWithInstances as rfi

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class Unloggedrfwi(rfi):

    def __init__(self, configspace, types, bounds, seed, **kwargs):
        """
        Interface to the random forest that takes instance features
        into account.

        Parameters
        ----------
        configspace: ConfigurationSpace
            configspace to be passed to random forest (used to impute inactive parameter-values)
        types: np.ndarray (D)
            Specifies the number of categorical values of an input dimension. Where
            the i-th entry corresponds to the i-th input dimension. Let say we have
            2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds: np.ndarray (D)
            Specifies the bounds
        seed: int
            The seed that is passed to the random_forest_run library.

        instance_features: np.ndarray (I, K)
            Contains the K dimensional instance features
            of the I different instances
        num_trees: int
            The number of trees in the random forest.
        do_bootstrapping: bool
            Turns on / off bootstrapping in the random forest.
        ratio_features: float
            The ratio of features that are considered for splitting.
        min_samples_split: int
            The minimum number of data points to perform a split.
        min_samples_leaf: int
            The minimum number of data points in a leaf.
        max_depth: int

        eps_purity: float

        max_num_nodes: int

        cutoff: int
            The cutoff used in the specified scenario

        threshold:
            Maximal possible value
        """
        super().__init__(configspace=configspace, types=types, bounds=bounds, seed=seed, **kwargs)

    # With the usage of pyrfr 0.8.0 this method is obsolete.
    # def _predict(self, X):
    #     """Predict means and variances for given X by first unlogging the leaf-values and then computing the mean for
    #     the trees NOT the training batch afterwards. The mean for the whole batch is handled by the parent class!
    #
    #     Parameters
    #     ----------
    #     X : np.ndarray of shape = [n_samples, n_features (config + instance
    #     features)]
    #
    #     Returns
    #     -------
    #     means : np.ndarray of shape = [n_samples, 1]
    #         Predictive mean
    #     vars : np.ndarray  of shape = [n_samples, 1]
    #         Predictive variance
    #     """
    #     if len(X.shape) != 2:
    #         raise ValueError(
    #             'Expected 2d array, got %dd array!' % len(X.shape))
    #     if X.shape[1] != self.types.shape[0]:
    #         raise ValueError('Rows in X should have %d entries but have %d!' %
    #                          (self.types.shape[0], X.shape[1]))
    #
    #     tree_mean_predictions = []
    #     tree_mean_variances = []
    #     for x in X:
    #         tmpx = np.array(list(map(lambda x_: np.power(10, x_), self.rf.all_leaf_values(x))))  # unlog values
    #         tree_mean_predictions.append(list(map(lambda x_: np.mean(x_), tmpx)))  # calculate mean and var
    #         tree_mean_variances.append(list(map(lambda x_: np.var(x_), tmpx)))  # over individual trees
    #
    #     mean = np.mean(tree_mean_predictions, axis=1)
    #     var = np.mean(tree_mean_variances, axis=1)
    #
    #     return mean.reshape((-1, 1)), var.reshape((-1, 1))

    def predict(self, X):
        """
        Method to override the predict method of RandomForestWithInstances.
        Thus it can be used in the marginalized over instances method of the RFWI class
        """
        return self._predict(X)
