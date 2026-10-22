# Export NVIDIA parakeet-tdt-0.6b-v3 to ONNX for parakeet-rs, or Moondream's Parakeet Ultra with
# --ultra.
#
# Parakeet Ultra (https://huggingface.co/moondream/parakeet-ultra) is a post-trained
# parakeet-tdt-0.6b-v3 published as a Hugging Face Transformers checkpoint. With --ultra, NVIDIA's
# model is loaded in NeMo and its weights are replaced with Moondream's. The weight names are
# mapped with the reverse of Transformers' own NeMo conversion (models/parakeet/convert_nemo_to_hf.py);
# it is a 1:1 rename. Moondream's small voice activity head, used only by their Photon runtime, is
# dropped.
#
# The encoder is exported with PyTorch's dynamo exporter (opset 23); onnx_export.py in this
# folder holds the shared settings, keep it next to this script.
#
# Setup:
#   pip install nemo_toolkit[asr] onnx onnxscript onnxruntime safetensors soundfile
#
# Usage:
#   python export_parakeet_tdt.py <output dir> [--check-audio speech.wav]
#   python export_parakeet_tdt.py <output dir> --ultra <parakeet-ultra dir>
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
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from safetensors import safe_open

import nemo.collections.asr as nemo_asr

import onnx_export

parser = argparse.ArgumentParser(description="Export Parakeet TDT 0.6B v3 (or Parakeet Ultra) to ONNX.")
parser.add_argument("output_dir", help="Where to write the ONNX files")
parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3", help="NVIDIA model name or .nemo path")
parser.add_argument("--ultra", help="Folder with Moondream's Parakeet Ultra model.safetensors and tokenizer.json")
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


def load_ultra(model, folder):
    if os.path.exists(os.path.join(folder, "ternary.json")):
        raise SystemExit("ternary checkpoints (Parakeet Redux) are not supported")
    weights = {}
    with safe_open(os.path.join(folder, "model.safetensors"), framework="np") as f:
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

    # The tokenizer must match the base model's, since vocab.txt comes from it.
    tokenizer = json.load(open(os.path.join(folder, "tokenizer.json")))
    vocab = tokenizer["model"]["vocab"]
    tokens = [t[0] for t in vocab] if isinstance(vocab, list) else sorted(vocab, key=vocab.get)
    if tokens[:len(model.tokenizer.vocab)] != list(model.tokenizer.vocab):
        raise SystemExit("tokenizer.json does not match the base model's vocabulary")


class Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length):
        return self.encoder(audio_signal=audio_signal, length=length)


print(f"loading {args.model}")
if args.model.endswith(".nemo"):
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
else:
    model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu")
model.eval()
if args.ultra:
    load_ultra(model, args.ultra)
    gc.collect()
onnx_export.use_sdpa(model.encoder)

# Example input (and the reference output for --check-audio).
if args.check_audio:
    audio, sr = sf.read(args.check_audio, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, "--check-audio must be 16 kHz mono"
else:
    audio = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32) * 0.1
features, feature_lengths = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)]))
encoder = Encoder(model.encoder)
expected = encoder(features, feature_lengths)[0].numpy()

os.makedirs(args.output_dir, exist_ok=True)
encoder_path = os.path.join(args.output_dir, "encoder-model.onnx")
with tempfile.TemporaryDirectory() as tmp:
    print("exporting the encoder")
    raw = os.path.join(tmp, "encoder.onnx")
    onnx_export.export(
        encoder,
        (features, feature_lengths),
        raw,
        input_names=["audio_signal", "length"],
        output_names=["outputs", "encoded_lengths"],
        dynamic_axes={"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"}},
    )
    onnx_export.save(onnx.load(raw), encoder_path, "encoder-model.onnx.data")

print("exporting the decoder/joint")
onnx_export.export_decoder_joint(model, os.path.join(args.output_dir, "decoder_joint-model.onnx"))

with open(os.path.join(args.output_dir, "vocab.txt"), "w") as f:
    for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
        f.write(f"{token} {i}\n")
del model
gc.collect()

session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
got = session.run(["outputs"], {"audio_signal": features.numpy(), "length": feature_lengths.numpy()})[0]
print(f"encoder ONNX vs NeMo: max abs diff {float(np.abs(got - expected).max()):.2e}")

for name in sorted(os.listdir(args.output_dir)):
    print(f"  {name}: {os.path.getsize(os.path.join(args.output_dir, name)) / 1e6:.0f} MB")
print(f"exported to: {args.output_dir}")
