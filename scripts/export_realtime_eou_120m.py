# Export NVIDIA parakeet_realtime_eou_120m-v1 (streaming ASR with end-of-utterance tokens) to
# ONNX for parakeet-rs.
#
# The encoder is exported as one cache-aware streaming step (cache_aware_stream_step), so the
# cache tensors are graph inputs and outputs. It uses PyTorch's dynamo exporter (opset 23);
# onnx_export.py in this folder holds the shared settings, keep it next to this script.
#
# Setup:
#   pip install nemo_toolkit[asr] onnx onnxscript onnxruntime tokenizers transformers
#
# Usage:
#   python export_realtime_eou_120m.py [model.nemo] [output_dir]
#   (without a .nemo path, nvidia/parakeet_realtime_eou_120m-v1 is downloaded)
#
# Output:
#   <output_dir>/
#     encoder.onnx        -- streaming encoder step
#     decoder_joint.onnx  -- RNNT decoder + joint network
#     tokenizer.json      -- tokenizer (Hugging Face tokenizers format)

import argparse
import logging
import os
import shutil
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, normalizers
from transformers.convert_slow_tokenizer import SentencePieceExtractor

import nemo.collections.asr as nemo_asr

import onnx_export

parser = argparse.ArgumentParser(description="Export parakeet_realtime_eou_120m-v1 to ONNX")
parser.add_argument("input_path", nargs="?", help=".nemo file (default: download from Hugging Face)")
parser.add_argument("output_dir", nargs="?", default="./onnx_eou", help="Directory for the ONNX files")
args = parser.parse_args()

# NeMo's typecheck wants keyword arguments; the exporter passes them by position.
logging.getLogger("nemo_logging").setLevel(logging.ERROR)
try:
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(False)
except ImportError:
    pass

if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir)

print(f"PyTorch version: {torch.__version__}")
nemo_path = args.input_path or hf_hub_download(
    "nvidia/parakeet_realtime_eou_120m-v1", "parakeet_realtime_eou_120m-v1.nemo"
)
model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
model.eval()
onnx_export.use_sdpa(model.encoder)

# --- tokenizer ---
# parakeet-rs reads tokenizer.json (Hugging Face tokenizers format), built from the SentencePiece
# BPE model in the .nemo. Its decoder keeps the "▁" word markers, like the published file.
with tempfile.TemporaryDirectory() as tmp:
    with tarfile.open(nemo_path, "r:*") as tar:
        member = next(m for m in tar.getnames() if m.endswith("tokenizer.model"))
        spm_path = os.path.join(tmp, "tokenizer.model")
        with open(spm_path, "wb") as out:
            out.write(tar.extractfile(member).read())
    spm = SentencePieceExtractor(spm_path).extract(models.BPE)
tokenizer = Tokenizer(models.BPE(vocab=spm["vocab"], merges=spm["merges"], unk_token="<unk>", fuse_unk=True))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.decoder = decoders.Metaspace(replacement=" ", prepend_scheme="always")
tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))
print("  tokenizer.json written")

# --- decoder/joint ---
print("Exporting decoder/joint...")
onnx_export.export_decoder_joint(model, os.path.join(args.output_dir, "decoder_joint.onnx"))

# --- streaming encoder step ---
print("Configuring the encoder for streaming...")
enc_cfg = model.cfg.encoder
subsampling = enc_cfg.get("subsampling_factor", 4)
chunk_frames = 16
if "streaming_cfg" in enc_cfg and enc_cfg.streaming_cfg.chunk_size:
    cs = enc_cfg.streaming_cfg.chunk_size
    chunk_frames = cs[1] if isinstance(cs, list) else cs
chunk = chunk_frames // subsampling
model.encoder.setup_streaming_params(chunk_size=chunk, shift_size=chunk)
model.encoder.streaming = True
drop_extra = getattr(model.encoder.streaming_cfg, "drop_extra_pre_encoded", True)


class StreamStep(torch.nn.Module):
    """One cache-aware encoder step with positional arguments."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        return self.encoder.cache_aware_stream_step(
            processed_signal=audio_signal,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra,
        )


audio = torch.randn(1, 16000)
features, feature_lengths = model.preprocessor(input_signal=audio, length=torch.tensor([16000]))
cache = model.encoder.get_initial_cache_state(batch_size=1)
step = StreamStep(model.encoder)
with torch.no_grad():
    reference = step(features, feature_lengths, *cache)[0].numpy()

encoder_path = os.path.join(args.output_dir, "encoder.onnx")
with tempfile.TemporaryDirectory() as tmp:
    raw_path = os.path.join(tmp, "encoder.onnx")
    print("Exporting the encoder...")
    onnx_export.export(
        step,
        (features, feature_lengths, *cache),
        raw_path,
        input_names=["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"],
        output_names=["outputs", "encoded_lengths", "new_cache_last_channel", "new_cache_last_time", "new_cache_last_channel_len"],
        dynamic_axes={"audio_signal": {2: "time"}, "outputs": {2: "time"}},
    )
    # 120M parameters: one file, no external data.
    onnx_export.save(onnx.load(raw_path), encoder_path)

session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
feeds = dict(zip([i.name for i in session.get_inputs()], [features.numpy(), feature_lengths.numpy(), *[c.numpy() for c in cache]]))
got = session.run(["outputs"], feeds)[0]
print(f"encoder ONNX vs NeMo: max abs diff {float(np.abs(got - reference).max()):.2e}")

for name in sorted(os.listdir(args.output_dir)):
    print(f"  {name}: {os.path.getsize(os.path.join(args.output_dir, name)) / 1e6:.1f} MB")
print(f"done: {args.output_dir}")
