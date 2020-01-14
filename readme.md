范例工程文件组织如下：
```
ExampleProject
├configs        #存放配置文件
│└test.yaml     #具体的配置文件
├main           #存放实验入口代码
│└some_exp.py   #包含一个run()函数
├src            #存放实验用的自己编写的代码
│├__init__.py
│└some_code.py
├utils          #存放管理实验用的工具的地方
│├__init__.py
│├argpaser.py   #管理实验的主要文件
│└logger.py     #用来记录日志和数据并保存
├results        #实验结果归档
│└...
└runner.py      #批量实验运行入口
```

其中结果记录文件夹组织如下：
```
results
└exp_record
 ├data
 │├seed_data.pkl
 │└seed_data.mat
 ├log
 │└seed_log.txt
 ├src
 │├__init__.py
 │└some_code.py
 ├config_old.yaml
 ├config.yaml
 └some_code.yaml
```
utils/argpaser.py的主要功能包括：
-读取原配置文件并归档
-解析命令行参数，覆盖配置文件并归档
-将源代码，实验入口文件归档

runner.py用于批量运行实验，针对seeds参数中的每一个种子运行一遍读取实验入口文件中的run()
