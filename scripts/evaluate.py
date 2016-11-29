from importance.importance.importance import Importance
from importance.utils.io.cmd_reader import CMDs

if __name__ == '__main__':
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()  # read cmd args
    importance = Importance(args.scenario_file, args.history, args.modus)  # create importance object
    print(importance.evaluator.to_evaluate)
