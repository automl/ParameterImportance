import numpy as np
from scipy import stats

from pimp.epm.base_epm import RandomForestWithInstances as rfi

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class EPARrfi(rfi):

    def __init__(self, configspace, types, bounds, seed,
                 cutoff=0,
                 threshold=0, **kwargs):
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
        np.seterr(divide='ignore', invalid='ignore')
        self.cutoff = cutoff
        self.threshold = threshold

    def _predict_EPAR(self, mean_var, prediction_threshold=0):
        """
        Predicting the Expected Penalized Average Runtime according to the cuttoff and par-factor specified in the
        scenario.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]
        """

        pred = np.zeros(shape=mean_var[0].shape)
        var = np.zeros(shape=mean_var[1].shape)

        for p in range(pred.shape[0]):
            var[p] = mean_var[1][p]
            if mean_var[0][p] > self.cutoff:
                # mean prediction is already higher than cutoff
                self.logger.debug("Predicted %g which is higher than cutoff"
                                  " %s" % (mean_var[0][p], self.cutoff))
                # pred[p] = tmp_threshold
                # continue

            # Calc cdf from -inf to cutoff
            cdf = stats.norm.cdf(x=self.cutoff, loc=mean_var[0][p],
                                 scale=np.sqrt(mean_var[1][p]))

            # Probability mass > cutoff
            upper_exp = 1 - cdf

            if upper_exp > 1:
                self.logger.warn("Upper exp is larger than 1, "
                                 "is this possible: %g > 1" % upper_exp)
                upper_exp = 1
                cdf = 0

            if upper_exp < prediction_threshold or self.threshold == self.cutoff:
                # There is not enough probability mass higher than cutoff
                # Or threshold == cutoff
                pred[p] = mean_var[0][p]
            else:
                # Calculate mean of lower truncnorm
                lower_pred = stats.truncnorm.stats(
                    a=(-np.inf - mean_var[0][p]) / np.sqrt(mean_var[1][p]),
                    b=(self.cutoff - mean_var[0][p]) / np.sqrt(mean_var[1][p]),
                    loc=mean_var[0][p],
                    scale=np.sqrt(mean_var[1][p]),
                    moments='m')

                upper_pred = upper_exp * self.threshold
                pred[p] = lower_pred * cdf + upper_pred

                if pred[p] > self.threshold + 10 ** -5:
                    raise ValueError("Predicted higher than possible, %g > %g"
                                     % (pred[p], self.threshold))

                # This can happen and if it happens, set prediction to cutoff
                if not np.isfinite(pred[p]):
                    self.logger.debug("Prediction is not finite cdf %g, "
                                      "lower_pred %g; Setting %g to %g" %
                                      (cdf, lower_pred, pred[p],
                                       self.cutoff + 10 ** -5))
                    pred[p] = self.cutoff + 10 ** -5
        return pred, var

    def predict(self, X):
        """
        Method to override the predict method of RandomForestWithInstances.
        Thus it can be used in the marginalized over instances method of the RFWI class
        """
        return self._predict_EPAR(self._predict(X))
