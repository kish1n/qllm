#!/usr/bin/env python3
"""Step 0 reference harness.

Loads an HF causal LM in fp32, runs a fixed prompt through it, and dumps
intermediate activations as raw float32 files plus a JSON manifest, so the
from-scratch C++ engine can be diffed against ground truth layer by layer.

Every architectural number (hidden size, layer count, head counts, RoPE
params, ...) is read from the model's config.json via AutoConfig -- nothing
here is hardcoded to Llama 3.2, so pointing --model at a SmolLM2 or Qwen
checkpoint should work without touching this script.

Usage:
    reference/.venv/bin/python reference/dump_golden.py \\
        --model /home/kish1n/models/unsloth-Llama-3.2-1B \\
        --out-dir reference/golden
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

FIXED_PROMPT = (
    "The quick brown fox jumps over the lazy dog, and then it ran to the "
    "river to drink some water."
)


def save_tensor(t: torch.Tensor, path: Path) -> list[int]:
    """Write a tensor as raw, row-major, little-endian float32 and return its shape."""
    t = t.detach().to(torch.float32).contiguous().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())
    return list(t.shape)


def extract_config_summary(config) -> dict:
    """Pull the numbers the C++ side needs out of config.json, generically."""
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    # Newer transformers folds rope_theta into rope_scaling/rope_parameters;
    # older ones (and other model families) may keep it as a top-level field.
    rope_scaling = (
        getattr(config, "rope_scaling", None)
        or getattr(config, "rope_parameters", None)
        or {}
    )
    rope_theta = None
    if isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta")
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)

    return {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        "head_dim": head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": getattr(config, "rms_norm_eps", None),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling if isinstance(rope_scaling, dict) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "golden"), help="output dir"
    )
    ap.add_argument("--prompt", default=FIXED_PROMPT)
    ap.add_argument("--gen-tokens", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading config for {args.model} ...")
    config = AutoConfig.from_pretrained(args.model)
    cfg_summary = extract_config_summary(config)
    print(json.dumps(cfg_summary, indent=2))

    print(f"[2/5] Loading tokenizer + model in fp32 on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(args.device)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    token_ids = input_ids[0].tolist()
    print(f"Prompt: {args.prompt!r}")
    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    manifest_tensors: dict[str, dict] = {}

    def record(name):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            shape = save_tensor(t, out_dir / f"{name}.bin")
            manifest_tensors[name] = {
                "file": f"{name}.bin",
                "shape": shape,
                "dtype": "float32",
            }

        return hook

    print("[3/5] Registering hooks ...")
    handles = [model.model.embed_tokens.register_forward_hook(record("embed_output"))]

    layer0 = model.model.layers[0]
    handles += [
        layer0.input_layernorm.register_forward_hook(record("layer0_attn_norm_output")),
        layer0.self_attn.q_proj.register_forward_hook(record("layer0_q_proj")),
        layer0.self_attn.k_proj.register_forward_hook(record("layer0_k_proj")),
        layer0.self_attn.v_proj.register_forward_hook(record("layer0_v_proj")),
        layer0.self_attn.register_forward_hook(record("layer0_attn_output")),
        layer0.mlp.register_forward_hook(record("layer0_mlp_output")),
    ]

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(record(f"layer{i}_output")))

    handles.append(model.model.norm.register_forward_hook(record("final_norm_output")))

    # Q/K after RoPE aren't exposed as a submodule -- apply_rotary_pos_emb is a
    # bare module-level function called by name inside attention.forward(), so
    # monkeypatch it to snapshot its first call (= layer 0, since layers run
    # in order). Resolved dynamically from the attention module's own file so
    # this keeps working for model families with their own modeling_*.py.
    model_module = importlib.import_module(type(layer0.self_attn).__module__)
    if not hasattr(model_module, "apply_rotary_pos_emb"):
        raise RuntimeError(f"{model_module.__name__} has no apply_rotary_pos_emb to patch")
    orig_rope = model_module.apply_rotary_pos_emb
    rope_captured = False

    def patched_rope(q, k, *rest, **kwargs):
        nonlocal rope_captured
        q_out, k_out = orig_rope(q, k, *rest, **kwargs)
        if not rope_captured:
            manifest_tensors["layer0_q_roped"] = {
                "file": "layer0_q_roped.bin",
                "shape": save_tensor(q_out, out_dir / "layer0_q_roped.bin"),
                "dtype": "float32",
            }
            manifest_tensors["layer0_k_roped"] = {
                "file": "layer0_k_roped.bin",
                "shape": save_tensor(k_out, out_dir / "layer0_k_roped.bin"),
                "dtype": "float32",
            }
            rope_captured = True
        return q_out, k_out

    model_module.apply_rotary_pos_emb = patched_rope

    print("[4/5] Running forward pass ...")
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits

    model_module.apply_rotary_pos_emb = orig_rope
    for h in handles:
        h.remove()

    if not rope_captured:
        raise RuntimeError("apply_rotary_pos_emb was never called; RoPE hook didn't fire")

    manifest_tensors["logits"] = {
        "file": "logits.bin",
        "shape": save_tensor(logits, out_dir / "logits.bin"),
        "dtype": "float32",
    }

    print(f"[5/5] Running greedy generate for {args.gen_tokens} tokens ...")
    with torch.no_grad():
        gen = model.generate(
            input_ids, max_new_tokens=args.gen_tokens, do_sample=False, num_beams=1
        )
    generated_ids = gen[0].tolist()

    manifest = {
        "model": args.model,
        "config": cfg_summary,
        "prompt": args.prompt,
        "token_ids": token_ids,
        "generated_ids": generated_ids,
        "generated_new_tokens": generated_ids[len(token_ids):],
        "tensors": manifest_tensors,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest_tensors)} tensors + manifest.json to {out_dir}")
    print("Generated text:", tokenizer.decode(generated_ids[len(token_ids):]))


if __name__ == "__main__":
    main()
