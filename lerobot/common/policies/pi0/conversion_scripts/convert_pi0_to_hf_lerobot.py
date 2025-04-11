import argparse
import pathlib

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from jax.sharding import SingleDeviceSharding

# ADDED: Import ml_dtypes to check type safely, though comparing name is often sufficient
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None # Handle case where ml_dtypes is not installed (less likely if JAX loads bf16)


from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.conversion_scripts.conversion_utils import (
    get_gemma_config,
    get_paligemma_config,
)
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

PRECISIONS = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


# ADDED: Helper function for safe conversion
def safe_numpy_to_torch(value, key_name="<unknown>"):
    """Converts numpy array to torch tensor, handling bfloat16."""
    if isinstance(value, np.ndarray):
        # Check for bfloat16 using dtype name for broader compatibility
        if value.dtype.name == 'bfloat16':
            # print(f"Converting key '{key_name}' from bfloat16 to float32") # Optional debug info
            return torch.from_numpy(value.astype(np.float32))
        else:
            # Try direct conversion for other supported numpy types
            try:
                return torch.from_numpy(value)
            except TypeError:
                print(f"Warning: Skipping key '{key_name}' due to unsupported numpy dtype: {value.dtype}")
                # Or raise the error if skipping is not desired
                # raise TypeError(f"Unsupported numpy dtype {value.dtype} for key '{key_name}'") from None
                return None # Indicate failure or skip
    elif isinstance(value, torch.Tensor):
         # If it's already a tensor, just return it
         return value
    else:
        # Handle non-ndarray cases (e.g., scalars) if necessary
        try:
            # General tensor creation might work for scalars
            return torch.tensor(value)
        except Exception as e:
            print(f"Warning: Skipping key '{key_name}'. Failed to convert value of type {type(value)} to tensor: {e}")
            return None # Indicate failure or skip


def slice_paligemma_state_dict(state_dict, config):
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # fmt: off
    # ... (rest of the slicing logic remains the same) ...
    state_dict["paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight"] = state_dict.pop(f"img/embedding/kernel{suffix}").transpose(
        3, 2, 0, 1
    )
    state_dict["paligemma.vision_tower.vision_model.embeddings.patch_embedding.bias"] = state_dict.pop(f"img/embedding/bias{suffix}")
    state_dict["paligemma.vision_tower.vision_model.embeddings.position_embedding.weight"] = state_dict.pop(f"img/pos_embedding{suffix}").reshape(
        -1, config.vision_config.hidden_size
    )
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")
    encoderblock_mlp_dense0_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")
    encoderblock_attention_0_key_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}")
    encoderblock_attention_0_key_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}")
    encoderblock_attention_0_value_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}")
    encoderblock_attention_0_value_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}")
    encoderblock_attention_0_query_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}")
    encoderblock_attention_0_query_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}")
    encoderblock_attention_0_out_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}")
    encoderblock_attention_0_out_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}")
    for i in range(config.vision_config.num_hidden_layers):
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
    state_dict["paligemma.vision_tower.vision_model.post_layernorm.weight"] = state_dict.pop(f"img/Transformer/encoder_norm/scale{suffix}").transpose()
    state_dict["paligemma.vision_tower.vision_model.post_layernorm.bias"] = state_dict.pop(f"img/Transformer/encoder_norm/bias{suffix}")
    state_dict['paligemma.multi_modal_projector.linear.weight'] = state_dict.pop(f"img/head/kernel{suffix}").transpose()
    state_dict['paligemma.multi_modal_projector.linear.bias'] = state_dict.pop(f"img/head/bias{suffix}")
    embedding_vector = state_dict.pop(f"llm/embedder/input_embedding{suffix}")
    state_dict["paligemma.language_model.model.embed_tokens.weight"] = embedding_vector
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")
    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")
    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")
    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].transpose(2, 0, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma.language_model.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma.language_model.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"paligemma.language_model.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]
    state_dict["paligemma.language_model.model.norm.weight"] = state_dict.pop(f"llm/final_norm/scale{suffix}")
    state_dict["paligemma.language_model.lm_head.weight"] = embedding_vector
    # fmt: on

    expert_dict = {}
    final_state_dict = {}
    keys_to_exclude = {  # Use a set for faster lookup
        f"llm/final_norm_1/scale{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
    }
    for key, value in state_dict.items():
        if key not in keys_to_exclude:
            # MODIFIED: Use the helper function for safe conversion
            torch_value = safe_numpy_to_torch(value, key_name=key)
            if torch_value is not None:
                 final_state_dict[key] = torch_value
        else:
            expert_dict[key] = value # Keep expert dict values as numpy for now (processed later)

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, num_expert=1):
    # fmt: off
    # ... (Gemma slicing logic remains the same) ...
    embedding_vector = torch.zeros([config.vocab_size, config.hidden_size])
    state_dict["gemma_expert.model.embed_tokens.weight"] = embedding_vector
    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")
    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")
    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")
    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        state_dict[f"gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].reshape(config.num_attention_heads * config.head_dim, config.hidden_size).transpose(1,0)
        state_dict[f"gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"gemma_expert.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]
    state_dict["gemma_expert.model.norm.weight"] = state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")
    state_dict["gemma_expert.lm_head.weight"] = embedding_vector
    # fmt: on

    final_state_dict = {}
    for key, value in state_dict.items():
        # MODIFIED: Use the helper function for safe conversion
        torch_value = safe_numpy_to_torch(value, key_name=key)
        if torch_value is not None:
            final_state_dict[key] = torch_value
        # The original code already handled existing torch.Tensors correctly,
        # our helper function does this too.

    return final_state_dict

# ... (flatten_for_memory, flatten_for_npz, slice_initial_orbax_checkpoint remain the same) ...
def flatten_for_memory(tree, parent_key=""):
    # ... (code as provided) ...
    out = {}
    for k, v in tree.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_memory(v, new_key))
        else:
            out[new_key] = np.array(v)  # Ensure conversion to np.array for consistency
    return out


def flatten_for_npz(tree, parent_key=""):
    # ... (code as provided) ...
    out = {}
    for k, v in tree.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_npz(v, new_key))
        else:
            # bf16/f32 here?
            out[new_key] = np.array(v)
    return out


def slice_initial_orbax_checkpoint(checkpoint_dir: str):
    # ... (code as provided) ...
    params_path = pathlib.Path(checkpoint_dir).resolve()
    checkpointer = ocp.PyTreeCheckpointer()

    metadata = checkpointer.metadata(params_path)
    print("Metadata keys:", list(metadata.keys()))

    params_name = "params"

    item = {params_name: metadata[params_name]}
    device = jax.local_devices()[0]  # Use the first local device
    sharding = SingleDeviceSharding(device)
    restored = checkpointer.restore(
        params_path,
        ocp.args.PyTreeRestore(
            item=item,
            restore_args=jax.tree_util.tree_map(
                lambda _: ocp.ArrayRestoreArgs(
                    restore_type=jax.Array,  # or np.ndarray, but bf16 is annoying about it
                    sharding=sharding,
                ),
                item,
            ),
            transforms={},
        ),
    )
    params = restored[params_name]

    # get params for PaliGemma
    pali_params = params["PaliGemma"]
    del params["PaliGemma"]
    pali_params_flat = flatten_for_npz(pali_params) # This converts JAX arrays to numpy arrays
    return {"paligemma_params": pali_params_flat, "projection_params": params} # projection_params might still contain JAX arrays or nested dicts


def update_keys_with_prefix(d: dict, prefix: str) -> dict:
    """Update dictionary keys by adding a prefix."""
    return {f"{prefix}{key}": value for key, value in d.items()}


def convert_pi0_checkpoint(checkpoint_dir: str, precision: str, tokenizer_id: str, output_path: str):
    # Break down orbax ckpts - they are in OCDBT
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir)
    # process projection params
    keys = [
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    projection_params = {}
    for key in keys:
        # MODIFIED: Need to handle potential nested dicts ('value') and ensure numpy conversion before safe_numpy_to_torch
        kernel_data = initial_params["projection_params"][key]["kernel"]
        bias_data = initial_params["projection_params"][key]["bias"]

        # Extract numpy array, handling the potential {'value': array} structure
        weight_np = np.array(kernel_data['value']) if isinstance(kernel_data, dict) else np.array(kernel_data)
        bias_np = np.array(bias_data['value']) if isinstance(bias_data, dict) else np.array(bias_data)

        # Convert safely to torch tensor
        torch_weight = safe_numpy_to_torch(weight_np, f"{key}.weight")
        torch_bias = safe_numpy_to_torch(bias_np, f"{key}.bias")

        if torch_weight is not None:
            projection_params[f"{key}.weight"] = torch_weight.T # Apply transpose after conversion
        if torch_bias is not None:
            projection_params[f"{key}.bias"] = torch_bias

    # Process PaliGemma weights
    paligemma_config = get_paligemma_config(precision)
    # paligemma_params are already torch tensors (or None if conversion failed), gemma_raw_dictionary contains numpy arrays
    paligemma_params, gemma_raw_dictionary = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )

    # Process Gemma weights (convert numpy arrays in gemma_raw_dictionary to torch tensors)
    gemma_config = get_gemma_config(precision)
    gemma_params = slice_gemma_state_dict(gemma_raw_dictionary, config=gemma_config) # gemma_params are torch tensors

    # Instantiate model from configs
    # ... (pi0_config selection logic remains the same) ...
    if "pi0_aloha_sim" in checkpoint_dir:
        pi0_config = PI0Config(
            empty_cameras=2,
            adapt_to_pi_aloha=True,
            use_delta_joint_actions_aloha=False,
        )
    elif "pi0_aloha_towel" in checkpoint_dir:
        pi0_config = PI0Config(
            adapt_to_pi_aloha=True,
            use_delta_joint_actions_aloha=True,
        )
    elif "pi0_base" in checkpoint_dir:
        pi0_config = PI0Config(
            empty_cameras=0,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
        )
    else:
        # ADDED: provide more context in the error
        raise ValueError(f"Could not determine PI0Config based on checkpoint_dir: {checkpoint_dir}")


    pi0_model = PI0Policy(pi0_config)

    paligemma_params = update_keys_with_prefix(paligemma_params, "model.paligemma_with_expert.")
    gemma_params = update_keys_with_prefix(gemma_params, "model.paligemma_with_expert.")
    projection_params = update_keys_with_prefix(projection_params, "model.")

    # load state dict
    torch_dtype = PRECISIONS[precision]
    # Combine all processed torch tensor dictionaries
    full_state_dict = {**paligemma_params, **gemma_params, **projection_params}

    # Filter out any None values that might have resulted from conversion errors
    filtered_state_dict = {k: v for k, v in full_state_dict.items() if v is not None}

    missing_keys, unexpected_keys = pi0_model.load_state_dict(filtered_state_dict, strict=False) # Use strict=False initially for debugging
    if missing_keys:
        print("Warning: Missing keys in state dict:", missing_keys)
    if unexpected_keys:
        print("Warning: Unexpected keys in state dict:", unexpected_keys)


    pi0_model = pi0_model.to(torch_dtype)
    # pi0_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id) # Tokenizer handling seems commented out

    print(f"Saving model to {output_path}...")
    pi0_model.save_pretrained(output_path, safe_serialization=True)
    # pi0_tokenizer.save_pretrained(output_path, dtype=torch_dtype) # Tokenizer handling seems commented out

    # assert that model loads properly
    print("Verifying model loading...")
    del pi0_model # Free memory
    _ = PI0Policy.from_pretrained(output_path)
    print("Model verification successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        # default="/raid/pablo/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params", # Keep default or remove if always required
        type=str,
        required=True, # Make it required if there's no sensible default
        help="Path to the ocdbt checkpoint directory (e.g., containing 'metadata', 'index')",
    )

    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        type=str,
        help="Precision identifier for model conversion. Should ideally match the base checkpoint precision for loading, but output will be PyTorch standard types.",
    )

    parser.add_argument(
        "--tokenizer_hub_id",
        default="google/paligemma-3b-pt-224",
        type=str,
        help="Hub path to the tokenizer to potentially save (currently commented out)",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save converted Hugging Face Lerobot weights to",
    )

    args = parser.parse_args()
    convert_pi0_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
        tokenizer_id=args.tokenizer_hub_id,
        output_path=args.output_path,
    )