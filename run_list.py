import time
import os
import argparse
from multiprocessing import Process
from multiprocessing import Manager


MAX_TASK_NUM = [4,4]
DEVICE_LIST = ['0','1']

m = Manager()
#task_number_list = m.list([MAX_TASK_NUM for i in DEVICE_LIST])
task_number_list = m.list([i for i in MAX_TASK_NUM])

def worker_function(cmd,device_idx,task_number_list):
    os.system('export CUDA_VISIBLE_DEVICES=%s;'%DEVICE_LIST[device_idx] + cmd)
    task_number_list[device_idx]+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run exp_list')
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    cmd_file = open(args.file, "r")
    cmd_list = cmd_file.readlines()
    cmd_file.close()
    cmd_line = 0
    worker_list = []
    while(True):
        cmd = cmd_list[cmd_line]
        if all([n==0 for n in task_number_list]):
            time.sleep(1)
            continue
        device_idx = task_number_list.index(max(task_number_list))
        task_number_list[device_idx]-=1
        p = Process(target=worker_function, args=(cmd.strip('\n'),device_idx,task_number_list))
        p.start()
        worker_list.append(p)
        cmd_line+=1
        if cmd_line == len(cmd_list):
            break
    for worker in worker_list:
        worker.join()