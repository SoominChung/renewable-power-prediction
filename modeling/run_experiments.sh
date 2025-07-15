#!/bin/bash

# 전체 실험을 GPU 5에서 실행
echo "Starting all experiments on GPU 5..."
nohup python modeling.py 5 > all_experiments.log 2>&1 &

echo "All experiments started in background on GPU 5"
echo "Monitor with: tail -f all_experiments.log"