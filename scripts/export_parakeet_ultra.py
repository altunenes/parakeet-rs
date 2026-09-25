# Export Moondream's Parakeet Ultra to ONNX for parakeet-rs.
#
# Parakeet Ultra (https://huggingface.co/moondream/parakeet-ultra) is a post trained
# parakeet-tdt-0.6b-v3 published as a Hugging Face Transformers checkpoint. This loads NVIDIA's
# parakeet-tdt-0.6b-v3 in NeMo, replaces its weights with Moondream's, and exports with NeMo's
# model.export(), the same way the TDT ONNX parakeet-rs already uses was made. The result is a
# drop in replacement for the `tdt/` folder.
#
# The weight names are mapped with the reverse of Transformers' own NeMo conversion
# (models/parakeet/convert_nemo_to_hf.py); it is a 1:1 rename. Moondream's small voice activity
# head, used only by their Photon runtime, is dropped.
#
# I ran this on Google Colab (CPU is enough; free Colab's RAM fits one model at a time).
#
# Colab setup:
#   !pip install nemo_toolkit[asr] onnx onnxruntime safetensors soundfile
#
# Usage:
#   python export_parakeet_ultra.py <parakeet-ultra dir> <output dir> [--check-audio speech.wav]
#
# <parakeet-ultra dir> holds Moondream's model.safetensors and tokenizer.json.
#
# Output:
#   <output_dir>/
#     encoder-model.onnx        -- encoder graph
#     encoder-model.onnx.data   -- encoder weights (external data, ~2.4 GB)
#     decoder_joint-model.onnx  -- TDT decoder + joint network
#     vocab.txt                 -- tokens, "<token> <id>" per line, "<blk>" last

import argparse
import gc
import json
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from safetensors import safe_open

import nemo.collections.asr as nemo_asr

parser = argparse.ArgumentParser(description="Export Moondream's Parakeet Ultra to ONNX (parakeet-rs TDT layout).")
parser.add_argument("input_dir", help="Folder with Moondream's model.safetensors and tokenizer.json")
parser.add_argument("output_dir", help="Where to write the ONNX files")
parser.add_argument("--base", default="nvidia/parakeet-tdt-0.6b-v3", help="NVIDIA base model (architecture + preprocessor)")
parser.add_argument("--check-audio", help="16 kHz mono WAV: compare the exported encoder with NeMo on it")
args = parser.parse_args()

torch.set_grad_enabled(False)

# Transformers name -> NeMo name, in this order (relative_k_proj contains "k_proj").
RULES = [
    (r"^encoder\.subsampling\.layers\.", "encoder.pre_encode.conv."),
    (r"^encoder\.subsampling\.linear\.", "encoder.pre_encode.out."),
    (r"^encoder\.encode_positions\.", "encoder.pos_enc."),
    (r"^(encoder\.layers\.\d+\.conv)\.norm\.", r"\1.batch_norm."),
    (r"\.relative_k_proj\.", ".linear_pos."),
    (r"\.q_proj\.", ".linear_q."),
    (r"\.k_proj\.", ".linear_k."),
    (r"\.v_proj\.", ".linear_v."),
    (r"\.o_proj\.", ".linear_out."),
    (r"\.bias_([uv])$", r".pos_bias_\1"),
    (r"^decoder\.embedding\.", "decoder.prediction.embed."),
    (r"^decoder\.lstm\.", "decoder.prediction.dec_rnn.lstm."),
    (r"^encoder_projector\.", "joint.enc."),
    (r"^decoder\.decoder_projector\.", "joint.pred."),
    (r"^joint\.head\.", "joint.joint_net.2."),
]


def to_nemo(name):
    for pattern, repl in RULES:
        name = re.sub(pattern, repl, name)
    return name


print(f"loading {args.base}")
model = nemo_asr.models.ASRModel.from_pretrained(args.base, map_location="cpu").eval()

# --- weights ---
if os.path.exists(os.path.join(args.input_dir, "ternary.json")):
    raise SystemExit("ternary checkpoints (Parakeet Redux) are not supported")
weights = {}
with safe_open(os.path.join(args.input_dir, "model.safetensors"), framework="np") as f:
    for key in f.keys():
        if key.startswith("vad_head."):
            continue
        value = f.get_tensor(key)
        # np.array keeps 0-d tensors 0-d; fp16 weights become fp32 like the base model
        weights[to_nemo(key)] = torch.from_numpy(np.array(value, dtype=np.float32 if value.dtype == np.float16 else value.dtype))

state = model.state_dict()
params = {name for name, _ in model.named_parameters()}
missing = sorted(params - weights.keys())
unexpected = sorted(weights.keys() - state.keys())
bad_shape = sorted(k for k in weights if k in state and weights[k].shape != state[k].shape)
if missing or unexpected or bad_shape:
    raise SystemExit(f"weight mapping failed: missing {missing[:5]}, unexpected {unexpected[:5]}, shape {bad_shape[:5]}")
state.update(weights)
model.load_state_dict(state, strict=True)
print(f"loaded {len(weights)} tensors into {len(params)} parameters "
      f"(kept from the base model: {sorted(state.keys() - weights.keys())})")
del state, weights
gc.collect()

# --- tokenizer must match the base model's, since vocab.txt comes from it ---
tokenizer = json.load(open(os.path.join(args.input_dir, "tokenizer.json")))
vocab = tokenizer["model"]["vocab"]
tokens = [t[0] for t in vocab] if isinstance(vocab, list) else sorted(vocab, key=vocab.get)
if tokens[:len(model.tokenizer.vocab)] != list(model.tokenizer.vocab):
    raise SystemExit("tokenizer.json does not match the base model's vocabulary")

# --- reference encoder output for the optional check (before the model is freed) ---
reference = None
if args.check_audio:
    audio, sr = sf.read(args.check_audio, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, "--check-audio must be 16 kHz mono"
    feats, flen = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)]))
    reference = (feats.numpy(), flen.numpy(), model.encoder(audio_signal=feats, length=flen)[0].numpy())

# --- export ---
os.makedirs(args.output_dir, exist_ok=True)
model.export(os.path.join(args.output_dir, "model.onnx"))  # -> encoder-model.onnx, decoder_joint-model.onnx
vocab_out = [*model.tokenizer.vocab, "<blk>"]
del model
gc.collect()

# Put all encoder weights in one external file, as parakeet-rs expects.
encoder_path = os.path.join(args.output_dir, "encoder-model.onnx")
encoder = onnx.load(encoder_path)
onnx.save(encoder, encoder_path, save_as_external_data=True, all_tensors_to_one_file=True,
          location="encoder-model.onnx.data", size_threshold=1024)
del encoder
keep = {"encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx", "vocab.txt"}
for name in os.listdir(args.output_dir):
    if name not in keep:
        os.remove(os.path.join(args.output_dir, name))

with open(os.path.join(args.output_dir, "vocab.txt"), "w") as f:
    for i, token in enumerate(vocab_out):
        f.write(f"{token} {i}\n")

for name in sorted(os.listdir(args.output_dir)):
    print(f"  {name}: {os.path.getsize(os.path.join(args.output_dir, name)) / 1e6:.0f} MB")

if reference is not None:
    feats, flen, expected = reference
    session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    got = session.run(["outputs"], {"audio_signal": feats, "length": flen})[0]
    print(f"encoder ONNX vs NeMo on {args.check_audio}: max abs diff {float(np.abs(got - expected).max()):.2e}")

print(f"exported to: {args.output_dir}")
