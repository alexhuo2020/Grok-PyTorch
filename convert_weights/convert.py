
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import os
import pickle
import torch 
import json
from safetensors.torch import save_file

## The 8-bit weight type, from xAI's code
@dataclass
class QuantizedWeight8bit:
    weight: jnp.array
    scales: jnp.array

    @property
    def shape(self):
        return self.weight.shape

def get_params(index, input_base_path):
    """
    index: index of the weight file: 0 -- 769
    input_base_path: the directory of the weights, e.g. ./checkpoints/ckpt-0
    """
    filename = os.path.join(input_base_path, f"tensor{index:05d}_000")
    with open(filename, "rb") as f:
        params = pickle.load(f)
    if hasattr(params,'scales'):
        params.weight =  params.weight
        params.scales =  params.scales
        try:
            params = params.weight * params.scales
        except:
            repeats = params.weight.shape[-2] // 8  ## The scales are of size 8,..., since it is sharded on 8 GPUs
            scales = jnp.repeat(params.scales, repeats, axis=-2)  
            params = params.weight.astype(scales.dtype) * scales

    params = np.array(params).astype(np.float32)
    return torch.tensor(params, dtype=torch.bfloat16).contiguous()



def convert_weights(input_base_path, output_path, params_partition_json_file):

    with open(params_partition_json_file, 'r') as f:
        modules_grok_index = json.load(f)

    for j in [0]: ## Total 64 layers of the model, the embedding and rms_norm weights are in the same file with the first layer
        weights = {}

        ## Mapping the weights name to the huggingface model weights name, convert 8-bit weights to bfloat16
        for param_name in modules_grok_index:
            if j == 0:
                if param_name == 'language_model/in_out_embed/embeddings':
                    weights['model.embed_tokens.weight'] = get_params(modules_grok_index[param_name], input_base_path)
                if param_name == 'language_model/rms_norm/scale':
                    weights['model.norm.weight'] = get_params(modules_grok_index[param_name], input_base_path)
            if param_name.startswith(f"transformer/decoder_layer_"):
                param_name_splits = param_name.split("/")
                layer_id = param_name_splits[1].split("_")[-1]
                if int(layer_id) == int(j):
                    if param_name_splits[-2] == 'query':
                        weights[f"model.layers.{layer_id}.self_attn.q_proj.weight"] = get_params(modules_grok_index[param_name], input_base_path).T
                    if param_name_splits[-2] == 'key':
                        weights[f"model.layers.{layer_id}.self_attn.k_proj.weight"] = get_params(modules_grok_index[param_name], input_base_path).T
                    if param_name_splits[-2] == 'value':
                        weights[f"model.layers.{layer_id}.self_attn.v_proj.weight"] = get_params(modules_grok_index[param_name], input_base_path).T
                    if param_name_splits[-2] == 'linear' and param_name_splits[-3]=='multi_head_attention':
                        weights[f"model.layers.{layer_id}.self_attn.o_proj.weight"] = get_params(modules_grok_index[param_name], input_base_path).T
                    if param_name_splits[-2] == 'rms_norm':
                        weights[f"model.layers.{layer_id}.input_layernorm.weight"] = get_params(modules_grok_index[param_name], input_base_path)
                    if param_name_splits[-2] == 'rms_norm_1':
                        weights[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = get_params(modules_grok_index[param_name], input_base_path)
                    if param_name_splits[-2] == 'rms_norm_2':
                        weights[f"model.layers.{layer_id}.pre_moe_layernorm.weight"] = get_params(modules_grok_index[param_name], input_base_path)
                    if param_name_splits[-2] == 'rms_norm_3':
                        weights[f"model.layers.{layer_id}.post_moe_layernorm.weight"] = get_params(modules_grok_index[param_name], input_base_path)
                    if param_name_splits[-2] == 'router':
                        weights[f"model.layers.{layer_id}.block_sparse_moe.gate.weight"] = get_params(modules_grok_index[param_name], input_base_path).T

                    if param_name_splits[-3] == 'moe':
                        p = get_params(modules_grok_index[param_name], input_base_path)
                        if param_name_splits[-2] == 'linear_v':
                            for i in range(8):
                                weights[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w3.weight"]  = p[i].T

                        if param_name_splits[-2] == 'linear':
                            print(modules_grok_index[param_name], input_base_path)
                            p = get_params(modules_grok_index[param_name], input_base_path)
                            for i in range(8):
                                weights[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w1.weight"]  = p[i].T

                        if param_name_splits[-2] == 'linear_1':
                            p = get_params(modules_grok_index[param_name], input_base_path)
                            for i in range(8):
                                weights[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w2.weight"]  = p[i].T    
        for key in weights:
            if not weights[key].is_contiguous():
                weights[key] = weights[key].contiguous()  # Make the tensor contiguous
        save_file(weights, f"{output_path}/model-{j:05}.safetensors", metadata={"format": "pt"})



        ## Get the weight -- filename mapping in json file
        weights = {}
        for param_name in modules_grok_index:
            if j == 0:
                if param_name == 'language_model/in_out_embed/embeddings':
                    weights['model.embed_tokens.weight'] = f"model-{j:05}.safetensors"
                if param_name == 'language_model/rms_norm/scale':
                    weights['model.norm.weight'] = f"model-{j:05}.safetensors"
            if param_name.startswith(f"transformer/decoder_layer_"):
                param_name_splits = param_name.split("/")
                layer_id = param_name_splits[1].split("_")[-1]
                if int(layer_id) == int(j):
                    if param_name_splits[-2] == 'query':
                        weights[f"model.layers.{layer_id}.self_attn.q_proj.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'key':
                        weights[f"model.layers.{layer_id}.self_attn.k_proj.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'value':
                        weights[f"model.layers.{layer_id}.self_attn.v_proj.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'linear' and param_name_splits[-3]=='multi_head_attention':
                        weights[f"model.layers.{layer_id}.self_attn.o_proj.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'rms_norm':
                        weights[f"model.layers.{layer_id}.input_layernorm.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'rms_norm_1':
                        weights[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'rms_norm_2':
                        weights[f"model.layers.{layer_id}.pre_moe_layernorm.weight"] = f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'rms_norm_3':
                        weights[f"model.layers.{layer_id}.post_moe_layernorm.weight"] =f"model-{j:05}.safetensors"
                    if param_name_splits[-2] == 'router':
                        weights[f"model.layers.{layer_id}.block_sparse_moe.gate.weight"] = f"model-{j:05}.safetensors"

                    if param_name_splits[-3] == 'moe':
                        if param_name_splits[-2] == 'linear_v':
                            for i in range(8):
                                weights[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w3.weight"]  = f"model-{j:05}.safetensors"

                        if param_name_splits[-2] == 'linear':
                            print(modules_grok_index[param_name], input_base_path)
                            for i in range(8):
                                weights[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w1.weight"]  = f"model-{j:05}.safetensors"

                        if param_name_splits[-2] == 'linear_1':
                            for i in range(8):
                                weights[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w2.weight"]  = f"model-{j:05}.safetensors"
    s = {}  
    for i in range(1):
        s.update({k: f"model-{i:05}.safetensors" for k, v in weights.items() if v == f"model-{i:05}.safetensors"})

    with open(f"{output_path}/model.safetensors.index.json", 'w') as f:
        json.dump({
        "metadata": {
            "total_size": 631294414632,
            "format": "pt"
        },
        "weight_map": 
            s
        }, f)



if __name__ == '__main__':

    input_base_path = "../checkpoints/ckpt-0"
    params_partition_json_file = "./params_partition.json"
    output_path = "../model_weights"

    convert_weights(input_base_path, output_path, params_partition_json_file)

