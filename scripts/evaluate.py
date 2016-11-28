from importance.importance.ablation import Ablation
from importance.importance.fanova import fANOVA
from importance.importance.forward_selection import ForwardSelector
from importance.utils.io.cmd_reader import CMDs
from importance.utils import Scenario, RunHistory2EPM4LogCost, RunHistory2EPM4Cost, RunHistory


if __name__ == '__main__':
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()



    evaluator = None
    if args.modus == 'ablation':
        evaluator = Ablation
    elif args.modus == 'fANOVA':
        evaluator = fANOVA
    else:
        evaluator = ForwardSelector
