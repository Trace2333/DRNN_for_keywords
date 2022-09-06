import os
import torch

def load_config(model, target_path, para_name, if_load_or_not):
    """
    读取对应的模型并装载
    Args:
        model: 模型
        target_path: 目标路径，指的是在check_points之后的路径，
        如"/RNN_attention/"
        if_load_or_not: True/False
    """
    if not os.path.exists("./check_points"):
        os.mkdir("./check_points")
    check_point = './check_points'
    if_load = if_load_or_not
    if if_load is True:
        if os.path.exists(check_point + target_path+ para_name):    # 参数加载
            model.load_state_dict(torch.load(check_point + target_path+ para_name))
            print(para_name + " Parms loaded!!!")
        elif not os.path.exists(check_point + target_path + para_name):
            #print(check_point + target_path + para_name + " is not exists!!!")
            raise RuntimeError(check_point + target_path + para_name + " is not exist.")
