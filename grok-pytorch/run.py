import torch 
torch.set_default_dtype(torch.bfloat16)
from modeling_grok import GrokForCausalLM 
from configuration_grok import GrokConfig
from transformers import BitsAndBytesConfig

config = GrokConfig()
config.num_hidden_layers = 1
config.architectures=  [
    "GrokForCausalLM"],
config.use_bfloat16 = True
config.torch_dtype=torch.bfloat16
import json
with open('../misc/output/config.json','w') as f:
    json.dump(config.to_dict(),f)

# model = GrokForCausalLM(config)

model = GrokForCausalLM.from_pretrained("../misc/output")
print(model.lm_head.weight)

from safetensors import safe_open
tensors = {}
with safe_open("../misc/output/model-00000.safetensors", framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
print(tensors['model.embed_tokens.weight'])

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16,
  # skip_modules=['lm_head']#, 'embed_tokens','norm']
)

model_nf4 = GrokForCausalLM.from_pretrained("../misc/output", quantization_config=nf4_config)
print(model_nf4)

model_nf4.save_pretrained('../grok4bit/')


