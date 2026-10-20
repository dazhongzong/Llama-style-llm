
import json
import struct
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path("models/SmolLM2-135M-Instruct")
)

config_path = model_dir / "config.json"
weight_path = model_dir / "model.safetensors"

if not config_path.is_file() or not weight_path.is_file():
    raise FileNotFoundError("缺少 config.json 或 model.safetensors")

config = json.loads(config_path.read_text(encoding="utf-8"))

with weight_path.open("rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_size))

weights = {
    name: info for name, info in header.items()
    if name != "__metadata__"
}

H = config["hidden_size"]
I = config["intermediate_size"]
L = config["num_hidden_layers"]
HQ = config["num_attention_heads"]
HKV = config["num_key_value_heads"]
D = H // HQ
V = config["vocab_size"]

expected = {
    "model.embed_tokens.weight": [V, H],
    "model.norm.weight": [H],
}

for i in range(L):
    p = f"model.layers.{i}."

    expected.update({
        p + "input_layernorm.weight": [H],
        p + "post_attention_layernorm.weight": [H],
        p + "self_attn.q_proj.weight": [HQ * D, H],
        p + "self_attn.k_proj.weight": [HKV * D, H],
        p + "self_attn.v_proj.weight": [HKV * D, H],
        p + "self_attn.o_proj.weight": [H, HQ * D],
        p + "mlp.gate_proj.weight": [I, H],
        p + "mlp.up_proj.weight": [I, H],
        p + "mlp.down_proj.weight": [H, I],
    })

if not config["tie_word_embeddings"]:
    expected["lm_head.weight"] = [V, H]

print("========== Model ==========")
print("Hidden Size:", H)
print("Layers:", L)
print("Query Heads:", HQ)
print("KV Heads:", HKV)
print("Head Dim:", D)
print("Vocab Size:", V)
print("RoPE Theta:", config["rope_theta"])
print("Weight file size:", weight_path.stat().st_size, "bytes")

print("\n========== Weight Validation ==========")

errors = []

for name, shape in expected.items():
    if name not in weights:
        errors.append(f"Missing: {name}")
        continue

    actual_shape = weights[name]["shape"]

    if actual_shape != shape:
        errors.append(
            f"Shape mismatch: {name}: "
            f"expected={shape}, actual={actual_shape}"
        )

print("Expected weight count:", len(expected))
print("Actual weight count:", len(weights))

dtype_counts = {}

for info in weights.values():
    dtype = info["dtype"]
    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

print("DTypes:", dtype_counts)

for name in [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
]:
    if name in weights:
        print(
            name,
            weights[name]["shape"],
            weights[name]["dtype"],
        )

extra = set(weights) - set(expected)
if extra:
    print("\nAdditional tensors:", sorted(extra)[:10])

if errors:
    print("\nValidation failed:")
    for err in errors:
        print(" -", err)
    raise SystemExit(1)

print("\n[PASS] Required weight names and shapes are correct")
