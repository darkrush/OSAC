import pickle
import sys
import datetime
import os
import scipy.io as io
import numpy

LOG_LEVEL = {'DEBUG':0,'INFO':10,'WARNING':20,'ERROR':30,'CRITIC':40}
LOG_TEXT = {'DEBUG':'[DEBUG]',
            'INFO':'[INFO]',
            'WARNING':'[WARNING]',
            'ERROR':'[ERROR]',
            'CRITIC':'[CRITIC]'}
COLOR_LOG_TEXT = {'DEBUG':'\033[1;36m[DEBUG]\033[0m',
                  'INFO':'\033[1;37m[INFO]\033[0m',
                  'WARNING':'\033[1;33m[WARNING]\033[0m',
                  'ERROR':'\033[1;31m[ERROR]\033[0m',
                  'CRITIC':'\033[1;31;43m[CRITIC]\033[0m'}
def meta2str(meta,color = False):
    if color:
        return "%s %s : %s"%(COLOR_LOG_TEXT[meta['level']],meta['time'],meta['info'])
    else:
        return "%s %s : %s"%(LOG_TEXT[meta['level']],meta['time'],meta['info'])

class Logger(object):
    def __init__(self):
        self.log_data_dict = {}
        self.log_info_list = []
        self.prefix = '0'
        self.output_dir = None
        self.log_level = 'INFO'

    def setup(self,output_dir):
        self.output_dir = output_dir
        os.mkdir(os.path.join(self.output_dir,'log'))
        os.mkdir(os.path.join(self.output_dir,'data'))

    def clean_up(self,prefix):
        self.prefix = prefix
        self.log_data_dict = {}
        self.log_info_list = []

    def set_log_level(self,log_level):
        assert log_level in LOG_LEVEL
        self.log_level = log_level

    #def get_dir(self):
    #    assert self.output_dir is not None
    #    return self.output_dir

    def add_log(self,log_info,level = 'INFO'):
        assert level in LOG_LEVEL
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        meta_info = {'time':time_str,'level':level,'info':log_info}
        self.log_info_list.append(meta_info)
        if LOG_LEVEL[self.log_level]<=LOG_LEVEL[level]:
            print(meta2str(meta_info,color = True))

    def append_data(self,name,x,y):
        assert type(x) in (int,float)
        if name not in self.log_data_dict:
            self.log_data_dict[name] = {'x':[],'y':[]}
        self.log_data_dict[name]['x'].append([x])
        self.log_data_dict[name]['y'].append([y] if type(y) in (int,float) else y)
    
    def dump_log(self):
        assert self.output_dir is not None
        with open(os.path.join(self.output_dir,'log',self.prefix+'_log.txt'),'w') as f:
            for meta_info in self.log_info_list:
                f.write(meta2str(meta_info)+'\n')
    def dump_data(self,dump_mat = False):
        assert self.output_dir is not None
        with open(os.path.join(self.output_dir,'data',self.prefix+'_data.pkl'),'wb') as f:
            pickle.dump(self.log_data_dict,f)
        if dump_mat:
            mat_dict = {}
            for key,value in self.log_data_dict.items():
                mat_dict[key+'_x']= numpy.stack(value['x'],axis = 0)
                mat_dict[key+'_y']= numpy.stack(value['y'],axis = 0)
            io.savemat(os.path.join(self.output_dir,'data',self.prefix+'_data.mat'),self.log_data_dict)
       


Singleton_logger = Logger()