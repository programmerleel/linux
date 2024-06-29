# -*- coding: utf-8 -*-
# @Time    : 2024/6/28/028 18:01
# @Author  : Shining
# @File    : config.py.py
# @Description :

import yaml

def load_config(config_path):
    file = open(config_path,"r",encoding="utf8")
    config = yaml.load(file,yaml.FullLoader)
    return config

if __name__ == '__main__':
    config = load_config(r"D:\project\HRNet-Pose-Segmentation\configs\base_config.yaml")
    print(config)