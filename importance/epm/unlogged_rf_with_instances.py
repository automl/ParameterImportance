import numpy as np
from scipy import stats
from importance.epm import RandomForestWithInstances

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class UnloggedRandomForestWithInstances(RandomForestWithInstances):

    def __init__(self, types, instance_features=None, num_trees=30, do_bootstrapping=True, n_points_per_tree=0,
                 ratio_features=5. / 6., min_samples_split=3, min_samples_leaf=3, max_depth=20, eps_purity=1e-8,
                 max_num_nodes=1000, seed=42, cutoff=0, threshold=0):
        """
        Interface to the random forest that takes instance features
        into account.

        Parameters
        ----------
        types: np.ndarray (D)
            Specifies the number of categorical values of an input dimension. Where
            the i-th entry corresponds to the i-th input dimension. Let say we have
            2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
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

        seed: int
            The seed that is passed to the random_forest_run library.

        cutoff: int
            The cutoff used in the specified scenario

        threshold:
            Maximal possible value
        """
        super().__init__(types=types, instance_features=instance_features, num_trees=num_trees,
                         do_bootstrapping=do_bootstrapping, n_points_per_tree=n_points_per_tree,
                         ratio_features=ratio_features, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                         eps_purity=eps_purity, max_num_nodes=max_num_nodes, seed=seed)
        self.cutoff = cutoff
        self.threshold = threshold

    def _unlogged_predict(self, X):
        """Predict means and variances for given X by first unlogging the leaf-values and then computing the mean for
        the trees NOT the training batch afterwards. The mean for the whole batch is handled by the parent class!

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0], X.shape[1]))

        tree_mean_predictions = []
        tree_mean_variances = []
        for x in X:
            tmpx = np.array(list(map(lambda x_: np.power(10, x_), self.rf.all_leaf_values(x))))  # unlog values
            tree_mean_predictions.append(list(map(lambda x_: np.mean(x_), tmpx)))  # calculate mean and var
            tree_mean_variances.append(list(map(lambda x_: np.var(x_), tmpx)))  # over individual trees

        mean = np.mean(tree_mean_predictions, axis=0)
        var = np.mean(tree_mean_variances, axis=0)

        return mean.reshape((-1, 1)), var.reshape((-1, 1))

    def _predict_EPAR(self, X, prediction_threshold=0):
        """
        Predicting the Expected Penalized Average Runtime according to the cuttoff and par-factor specified in the
        scenario.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]
        """

        mean_var = self._unlogged_predict(X=X)

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
                    self.logger.critical("Prediction is not finite cdf %g, "
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
        return self._predict_EPAR(X)
