#!/bin/bash
#bash:/build.sh:/bin/bash^M:bad interpreter:No such file or directory报错解决方法
#1.  vi build.sh
#2.  :set ff
#3.  :set fileformat=unix
#4.  :wq
export DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=172.17.0.2 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=1
export PYTHONPATH="/root/singa/build/python/"
python3 mnist_cnn.py