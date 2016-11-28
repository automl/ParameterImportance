from importance.importance.ablation import Ablation
from importance.importance.fanova import fANOVA
from importance.importance.forward_selection import ForwardSelector
from importance.utils.io.cmd_reader import CMDs
from importance.utils.io.input import InputHandler
from importance.epm import RandomForestWithInstances


if __name__ == '__main__':
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()
    ih = InputHandler(args.scenario_file, args.history)  # Read all the inputs
    model = RandomForestWithInstances(ih.types).train(ih.X, ih.y)
    parameters_to_evaluate = []  # TODO Find way of allowing for users to specify which parameters to evaluate

    evaluator = None
    if args.modus == 'ablation':
        evaluator = Ablation(ih.scenario.cs, model, parameters_to_evaluate)
    elif args.modus == 'fANOVA':
        evaluator = fANOVA(ih.scenario.cs, model, parameters_to_evaluate)
    else:
        evaluator = ForwardSelector(ih.scenario.cs, model, parameters_to_evaluate)
    print(str(evaluator))
