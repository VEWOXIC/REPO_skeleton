"""
cfg system: 还没写

用户提供 模型, dataset, cfg名称(非必要), 一些参数设置(非必要)
系统读取 cfgs/exps/模型/模型_dataset_cfg名称.json

为default, 未提供cfg名称时
系统读取 cfg名称默认cfgs/exps/模型/模型_dataset_default.json, 不存在的话自动创建

用户提供的参数用于inplace修改cfg内容, 仅针对一次实验, 不保存

cfg结合exp,model,data为一个整体, 传入模型和exp.py, 书写格式参考cfgs/exps/basic.json

1:  cfgs/exps/basic.json 提供详细注释和说明
2:  exp文件我还没修改完, 想改的内容写注释了
"""