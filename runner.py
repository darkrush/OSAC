import os
import importlib.util
import random
import numpy
import torch
from utils.argpaser import Singleton_argpaser as args
from utils.logger import Singleton_logger as logger
logger.setup(args.exp_dir)
logger.set_log_level('CRITIC')

if torch.cuda.is_available():
    # ensure deterministic of experiment running
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

spec = importlib.util.spec_from_file_location(os.path.basename(args.main),args.main)
experiment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_module)
for seed_idx in range(len(args.seeds)):
    seed = args.seeds[seed_idx]
    logger.clean_up(prefix = str(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    experiment_module.run()
    print('Exp process %d/%d'%(seed_idx,len(args.seeds)))
    