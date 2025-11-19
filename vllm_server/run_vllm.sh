#!/bin/bash

docker run --gpus all -p 8000:8000 \
  -v ~/yong/RLN/models/llama3-8b:/model \
  --name llama31_8b_vllm \
  --rm \
  -d \
  llama31_8b_vllm