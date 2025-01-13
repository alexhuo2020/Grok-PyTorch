## Convert original model weights to safetensors

The original model weights provided by xAI are quantized in 8-bit. We convert them to `bfloat16` type for easy use with HuggingFace framework. The original model weights files only contain the model weight and no further information is given. Hence we need to first figure out how the files are related to the model's modules.

## Get parameters partitions
1. First run 
    ```bash
    git clone github.com/xai-org/grok-1
    ```
    to get the official code by xAI.

2. Then copy `test.py` in to the folder and run `test.py` without using GPU (also modify the `runners.py` line 215 to         `from_checkpoint: bool = False`). Since JAX provided support for manually creating a device map with 
    ```python
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
    ``` 
    However, due to the large memory requirement, we might not want to run the code in CPU. Runing `python test.py` will get message in `error.txt`, which contains the information of the parameter sharding states.

3. Run `python params_partition.py` and we will get the `params_partition.csv` file, for example, its first few rows:
    ```csv
    ,name,dim
    0,language_model/in_out_embed/embeddings,"[131072, 6144]"
    1,language_model/rms_norm/scale,[6144]
    2,transformer/decoder_layer_0/moe/linear/w,"[8, 6144, 32768]"
    3,transformer/decoder_layer_0/moe/linear_1/w,"[8, 32768, 6144]"
    ```
    This gives us information on the index of the model weight files and the name of the corresponding module of the weights. Moreover, we can also get the shape of the module weights.

## Unquantize the model weights and convert to safetensors
4. Run `python convert.py` to convert the model weights to torch tensors in `bfloat16` format. The result is in the `model_weights` folder and contains 
    - `model.safetensors.index.json`: information of the modules and the corresponding file index
    - various `model-00....safetensors`: the weights in `bfloat16` format.


