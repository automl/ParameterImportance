from importance.evaluator.ablation import Ablation
from importance.evaluator.fanova import fANOVA

from importance.epm import RandomForestWithInstances
from importance.evaluator.forward_selection import ForwardSelector
from importance.importance.importance import Importance
from importance.utils.io.cmd_reader import CMDs

if __name__ == '__main__':
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()
    ih = Importance(args.scenario_file, args.history)  # Read all the inputs
    model = RandomForestWithInstances(ih.types).train(ih.X, ih.y)
    parameters_to_evaluate = []  # TODO Find way of allowing for users to specify which parameters to evaluate

    evaluator = None
    if args.modus == 'ablation':
        evaluator = Ablation(ih.scenario.cs, model, parameters_to_evaluate)
    elif args.modus == 'fANOVA':
        evaluator = fANOVA(ih.scenario.cs, model, parameters_to_evaluate)
    else:
        evaluator = ForwardSelector(ih.scenario.cs, model, parameters_to_evaluate)
    print(evaluator.X)
