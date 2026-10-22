# Export NVIDIA parakeet-ctc-0.6b to ONNX for parakeet-rs.
#
# One graph, encoder plus CTC head. Its interface is the Transformers one parakeet-rs reads:
# input_features [batch, frames, 80] and attention_mask [batch, frames] in, logits
# [batch, frames / 8, vocab + 1] out (raw logits; blank is the last id). tokenizer.json is
# NVIDIA's own, from the Hugging Face model repo.
#
# The graph is exported with PyTorch's dynamo exporter (opset 23); onnx_export.py in this
# folder holds the shared settings, keep it next to this script.
#
# Setup:
#   pip install nemo_toolkit[asr] onnx onnxscript onnxruntime
#
# Usage:
#   python export_parakeet_ctc.py <output dir> [--model nvidia/parakeet-ctc-0.6b or a .nemo path]
#
# Output:
#   <output_dir>/
#     model.onnx        -- encoder + CTC head
#     model.onnx_data   -- weights (external data, ~2.4 GB)
#     tokenizer.json

import argparse
import os
import shutil
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download

import nemo.collections.asr as nemo_asr

import onnx_export

parser = argparse.ArgumentParser(description="Export Parakeet CTC 0.6B to ONNX.")
parser.add_argument("output_dir", help="Where to write the ONNX files")
parser.add_argument("--model", default="nvidia/parakeet-ctc-0.6b", help="NVIDIA model name or .nemo path")
args = parser.parse_args()

torch.set_grad_enabled(False)


class CtcModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.head = model.decoder.decoder_layers

    def forward(self, input_features, attention_mask):
        length = attention_mask.sum(-1)
        encoded, _ = self.encoder(audio_signal=input_features.transpose(1, 2), length=length)
        return self.head(encoded).transpose(1, 2)


print(f"loading {args.model}")
if args.model.endswith(".nemo"):
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
else:
    model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu")
model.eval()
onnx_export.use_sdpa(model.encoder)

audio = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32) * 0.1
features, feature_lengths = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)]))
features = features.transpose(1, 2).contiguous()
mask = torch.ones(features.shape[:2], dtype=torch.int64)
ctc = CtcModel(model)
expected = ctc(features, mask).numpy()

os.makedirs(args.output_dir, exist_ok=True)
model_path = os.path.join(args.output_dir, "model.onnx")
with tempfile.TemporaryDirectory() as tmp:
    raw = os.path.join(tmp, "model.onnx")
    onnx_export.export(
        ctc,
        (features, mask),
        raw,
        input_names=["input_features", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
    )
    onnx_export.save(onnx.load(raw), model_path, "model.onnx_data")

repo = args.model if not args.model.endswith(".nemo") else "nvidia/parakeet-ctc-0.6b"
shutil.copy(hf_hub_download(repo, "tokenizer.json"), os.path.join(args.output_dir, "tokenizer.json"))

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
got = session.run(["logits"], {"input_features": features.numpy(), "attention_mask": mask.numpy()})[0]
print(f"ONNX vs NeMo: max abs diff {float(np.abs(got - expected).max()):.2e}")

for name in sorted(os.listdir(args.output_dir)):
    print(f"  {name}: {os.path.getsize(os.path.join(args.output_dir, name)) / 1e6:.0f} MB")
print(f"exported to: {args.output_dir}")
