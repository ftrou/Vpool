# inference.py
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# -----------------------------------------------------------------------------
# Environment & Model Setup
# -----------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model_name = "meta-llama/Meta-Llama-3.1-405B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model with device_map='auto' and offload folder...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload"
)
model.eval()

# -----------------------------------------------------------------------------
# Inference Function Using Mixed Precision
# -----------------------------------------------------------------------------
def run_inference(model, input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to("cuda")
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_length=200
            )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------------------------------------------------------
# System Performance Monitoring Functions
# -----------------------------------------------------------------------------
def monitor_system_performance():
    gpu_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
    mem = psutil.virtual_memory()
    print(f"GPU VRAM Usage: {gpu_memory:.2f} GB")
    print(f"System RAM Usage: {mem.percent}% ({mem.used/(1024 ** 3):.2f} GB used out of {mem.total/(1024 ** 3):.2f} GB)")

def final_performance_report():
    gpu_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
    mem = psutil.virtual_memory()
    print(f"\nFinal GPU VRAM usage: {gpu_memory:.2f} GB")
    print(f"Final System RAM usage: {mem.percent}%")

# -----------------------------------------------------------------------------
# Main Inference and Performance Monitoring
# -----------------------------------------------------------------------------
input_text = "Explain how photosynthesis works in plants."

print("Running inference...")
start_inference_time = time.time()
output = run_inference(model, input_text)
end_inference_time = time.time()

print(f"Inference time: {end_inference_time - start_inference_time:.2f} seconds")
monitor_system_performance()

print("Model output:")
print(output)

final_performance_report()
