#!/usr/bin/env bash
source /Users/wanqianjun/tensorflow/bin/activate
zip -r train.zip *.py
python upload_to_oss.py