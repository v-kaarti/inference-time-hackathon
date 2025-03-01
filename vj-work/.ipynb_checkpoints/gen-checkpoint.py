from vllm import LLM, SamplingParams
import torch

prompts = [
    r"What is the capital of France? Think thoroughly and then answer. You must begin your output with <think>\n"
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95
)

# Configure for 8 H100s
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    tensor_parallel_size=8,
    max_model_len=1024,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    dtype=torch.bfloat16,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")