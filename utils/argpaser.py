import argparse
import datetime
import yaml
import sys
import os
import shutil

SRC_DIR = 'src'
DEFAULT_RESULTS_DIR = 'results'
OLD_CONFIG_FILE = 'config_old.yaml'
NEW_CONFIG_FILE = 'config.yaml'

class ArgPaser(object):
    def __init__(self):
        local_argv = sys.argv[1:]
        assert local_argv.count('--config') + local_argv.count('-C') is 1
        config_flag = '--config' if '--config' in local_argv else '-C'
        config_file = local_argv.pop(local_argv.index(config_flag)+1)
        local_argv.remove(config_flag)
        with open(config_file,'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        parser = argparse.ArgumentParser(description='Experiment Configs & Arguments parser')
        for key,value in yaml_dict.items():
            parser.add_argument('--'+key,default = value, type=type(value))
        main_file = local_argv.pop(0)
        print(main_file)
        assert os.path.exists(main_file)
        parser.add_argument('--main',default = main_file, type=str)
        default_exp_dir = os.path.basename(main_file).split('.')[0]+datetime.datetime.now().strftime('_%Y-%m-%d_%H:%M:%S')
        full_result_dir = os.path.join(DEFAULT_RESULTS_DIR,default_exp_dir)
        parser.add_argument('--exp-dir',default = full_result_dir, type=str)

        for item_index in range(len(local_argv)):
            if local_argv[item_index].startswith('--task_coef') \
               and local_argv[item_index].lstrip('--') not in yaml_dict.keys() :
                parser.add_argument(local_argv[item_index],default = float(local_argv[item_index+1]), type=float)

        args = parser.parse_args(local_argv)
        self.__dict__.update(args.__dict__)
        full_result_dir = args.exp_dir
        os.makedirs(full_result_dir, exist_ok=False)
        shutil.copy(config_file,os.path.join(full_result_dir,OLD_CONFIG_FILE))
        with open(os.path.join(full_result_dir,NEW_CONFIG_FILE), 'w') as yaml_file:
            yaml.dump(vars(args), yaml_file,default_flow_style=False,encoding='utf-8',allow_unicode=True)
        shutil.copy(main_file,args.exp_dir)
        shutil.copytree(SRC_DIR, os.path.join(args.exp_dir,SRC_DIR), ignore=shutil.ignore_patterns('__pycache__'))

Singleton_argpaser = ArgPaser()
