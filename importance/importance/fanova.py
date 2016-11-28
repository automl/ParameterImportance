from importance.importance.base_evaluator import AbstractEvaluator


class fANOVA(AbstractEvaluator):

    def run(self):
        raise NotImplementedError
        # fanova.get_marginal list dimension list / position of parameters in configspace to analyze

        # for parameter in self.to_evaluate:
        #     get_idx in ordered_dict_of_config_space
        #     imp_val_of_param[parameter] = fanova.get_marginal(dim_list=[idx])
