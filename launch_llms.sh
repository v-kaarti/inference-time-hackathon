#!/bin/bash
# Launch the DeepSeek-R1-Distill-Llama-8B model server
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6 \
    --dtype bfloat16 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 512 \
    --enforce_eager &

# Launch the DeepSeek-R1-Distill-Qwen-1.5B model server on port 8001
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --tensor-parallel-size 4 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.2 \
    --dtype bfloat16 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 512 \
    --enforce_eager \
    --port 8001 &