from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os
import torch
import transformers.utils.import_utils


if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):     #fix is_torch_fx_available error
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

base_model_path = os.path.expanduser("~/Scratch/open_source_llms/deepseek-moe-16b-chat")
new_model_path = os.path.expanduser("~/Scratch/open_source_llms/deepseek-moe-16b-chat-masked")

# create a folder
os.makedirs(new_model_path, exist_ok=True)

print("Loading config and sanitizing RoPE scaling...")
config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    scaling_type = config.rope_scaling.get("type", None)
    if scaling_type == "default" or scaling_type not in ["linear", "dynamic"]:
        config.rope_scaling["type"] = "linear"
        if "factor" not in config.rope_scaling:
            config.rope_scaling["factor"] = 1.0

print("Loading tokenizer and model (this may take a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    config=config, 
    trust_remote_code=True, 
    device_map="cpu"
)

#add [MASK]
special_tokens_dict = {'additional_special_tokens': ['[MASK]']}
tokenizer.add_special_tokens(special_tokens_dict)
print(f"[MASK] token id: {tokenizer.convert_tokens_to_ids('[MASK]')}")

# adjust embedding layer
print("Resizing token embeddings...")
model.resize_token_embeddings(len(tokenizer))

#using the underlying PyTorch save method to bypass transformer bugs
print(f"Manually saving model components to {new_model_path}...")
#save config
config.save_pretrained(new_model_path)
#save tokenizer
tokenizer.save_pretrained(new_model_path)

#save the full weight with torch.save
torch.save(model.state_dict(), os.path.join(new_model_path, "pytorch_model.bin"))

print("Done! Masked model successfully created.")