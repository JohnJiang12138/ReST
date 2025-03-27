'''
Author: zhangle09 zhangle09@baidu.com
Date: 2024-05-09 17:19:27
LastEditors: zhangle09 zhangle09@baidu.com
LastEditTime: 2024-05-09 17:46:13
FilePath: /augment/server/client.py
'''
import requests
import json
import os

# Flask 应用的 URL 和端点
url = 'http://127.0.0.1:5000/process'

current_file_path = os.path.abspath(__file__)
current_dir = '/'.join(os.path.dirname(current_file_path).split('/')[:-1])
# current_dir = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReST"
print(current_dir)

# 输入和输出文件夹路径的 JSON 数据
data = {
    'input_path': current_dir + '/data/input/',
    'output_path': current_dir + '/data/output/',
    'prompt': "Translate German to English: "
}

# 发送 POST 请求
response = requests.post(url, json=data)

# 打印服务器响应的 JSON 数据
print('Status Code:', response.status_code)
print('Response JSON:', response.json())