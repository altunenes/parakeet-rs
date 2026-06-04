# Export NVIDIA Nemotron 3.5 Streaming ASR (multilingual, 0.6B) to ONNX
#
# This is the multilingual sibling of export_nemotron_streaming.py. The model
# itself is "Nemotron-3.5-ASR-Streaming-Multilingual-0.6b" -- same FastConformer
# backbone as the English-only 0.6B, plus a prompt MLP that conditions decoding
# on a language id (en-US, ja-JP, ...).
#
# A few things I had to figure out the hard way and want to remember:
#
# (1) Earlier on, when the model was still WIP, the EncDecRNNTBPEModelWithPrompt
#     class only lived on a contributor fork. That branch has since been merged
#     into NVIDIA/NeMo@main, so the install is back to the boring one-liner:
#
#       !pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
#       !pip install 'numpy<2.0' Cython packaging onnx onnxruntime soundfile sentencepiece -q
#
#     Then Runtime -> Restart session
#
# (2) The encoder graph itself is structurally identical to the English-only
#     model. cache_aware_stream_step has the same signature. The only real
#     differences are:
#       - att_context_size default is [56, 0] here (vs [70, 6] for English).
#         I export with [56, 6] so the chunk granularity (valid_out_len=7)
#         matches the English path and I don't have to retune the streaming
#         buffer logic on the Rust side. The model card officially supports
#         right-contexts {0, 1, 3, 6, 13} corresponding to 80/160/320/560/1120 ms.
#       - last_channel_cache_size becomes 56 (was 70) -- one constant change.
#
# (3) The "prompt" lives entirely outside the encoder. It's a tiny MLP
#     (Linear 1152 -> ReLU -> Linear 1024) that NeMo applies AFTER the cache-
#     aware step on the encoder output. The "1152" decomposes as 1024 (encoder
#     hidden) + 128 (one-hot language vector). NeMo stores the chosen lang as
#     `self._inference_prompt_index` -- just a Python int -- which means the
#     graph would bake one language as a constant if I just traced it.
#
#     My wrapper below exposes `prompt_index` as a real ONNX input instead, so
#     one ONNX serves every language. The Rust side just looks up the index
#     from the prompt_dictionary and feeds it.
#
# (4) The .nemo carries a `prompt_dictionary` mapping language codes to ids
#     in [0, 128). I dump it into config.json next to the ONNX files. The
#     Rust loader doesn't strictly need this file (it has the dictionary
#     embedded), but external consumers can use it as the source of truth.
#
# (5) I use NeMo's `pad_and_drop_preencoded=True` mode on the streaming buffer.
#     That's the path NVIDIA themselves recommend for ONNX export. it makes
#     the per-chunk layout uniform across step 0 and subsequent steps, so
#     `drop_extra_pre_encoded=2` can be baked into the graph as a constant
#     instead of needing to vary per call.
#
# Output layout matches the English script (single encoder.onnx + .data, one
# decoder_joint.onnx, plus tokenizer.model and config.json), so the Rust
# auto-detection in src/nemotron.rs picks the multilingual variant up just
# by spotting the `prompt_index` input on the encoder graph.
#
# Colab usage:
#   python export_nemotron_streaming_multilingual.py \
#       nemotron-3.5-asr-streaming-0.6b.nemo ./onnx_nemotron_multilingual
#
# Output:
#   <output_dir>/
#     encoder.onnx          -- streaming encoder + prompt-kernel head
#     encoder.onnx.data     -- encoder weights (external data, ~2.5 GB)
#     decoder_joint.onnx    -- RNNT decoder + joint network
#     tokenizer.model       -- SentencePiece tokenizer (vocab ~13k, includes lang tags)
#     config.json           -- streaming params, cache shapes, prompt_dictionary

import argparse
import functools
import gc
import glob
import json
import logging
import os
import shutil
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Export Nemotron 3.5 streaming ASR model (multilingual) to ONNX"
)
parser.add_argument("input_path", help="Path to .nemo model file")
parser.add_argument("output_dir", help="Directory for ONNX outputs")
parser.add_argument(
    "--left-context", type=int, default=56,
    help="Attention left context (default: 56 -- the only value this model supports)",
)
parser.add_argument(
    "--right-context", type=int, default=6,
    help="Attention right context (default: 6; valid values per model card are 0, 3, 6, 13)",
)
parser.add_argument(
    "--verify-langs", default="en-US,es-ES,ja-JP",
    help="Comma-separated language codes to verify ONNX output matches NeMo for",
)
args = parser.parse_args()

ATT_CONTEXT_SIZE = [args.left_context, args.right_context]
OUTPUT_DIR = args.output_dir
VERIFY_LANGS = [s.strip() for s in args.verify_langs.split(",") if s.strip()]

# ---------------------------------------------------------------------------
# NeMo is extremely chatty; mute most of it
# ---------------------------------------------------------------------------

logging.getLogger("nemo_logging").setLevel(logging.ERROR)
try:
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# PyTorch 2.9+ defaults torch.onnx.export to the dynamo path, which silently
# breaks tracing for this model. I force it back to the legacy exporter.
# Same shim I use in the English-only script.
# ---------------------------------------------------------------------------

pytorch_version = tuple(
    int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
)
print(f"PyTorch version: {torch.__version__}")

_PATCH_MARKER = "_legacy_onnx_patched"
if pytorch_version >= (2, 9) and not getattr(torch.onnx.export, _PATCH_MARKER, False):
    print("  Patching torch.onnx.export for PyTorch 2.9+ (dynamo=False)")
    _original_torch_onnx_export = torch.onnx.export

    @functools.wraps(_original_torch_onnx_export)
    def _patched_onnx_export(*pargs, **kwargs):
        if "dynamo" not in kwargs:
            kwargs["dynamo"] = False
        return _original_torch_onnx_export(*pargs, **kwargs)

    _patched_onnx_export._legacy_onnx_patched = True
    torch.onnx.export = _patched_onnx_export

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"\nLoading model from: {args.input_path}")

model = nemo_asr.models.ASRModel.restore_from(args.input_path, map_location=device)
model.eval()

print(f"  Model class : {type(model).__name__}")
print(f"  Encoder type: {type(model.encoder).__name__}")
print(f"  Vocab size  : {model.tokenizer.vocab_size}")
print(f"  num_prompts : {model.num_prompts}")

# Sanity check: this script is only meaningful for the prompted variant.
# If somebody points it at the English-only .nemo it would silently produce
# a single-language ONNX with prompt_index baked to 0, which is wrong.
if type(model).__name__ != "EncDecRNNTBPEModelWithPrompt":
    raise RuntimeError(
        f"Expected EncDecRNNTBPEModelWithPrompt, got {type(model).__name__}. "
        "Use export_nemotron_streaming.py for the English-only model."
    )

# ---------------------------------------------------------------------------
# Extract tokenizer (SentencePiece .model file, embedded in the .nemo tar)
# ---------------------------------------------------------------------------

print("\nExtracting tokenizer...")
with tarfile.open(args.input_path, "r:*") as tar:
    for member in tar.getnames():
        if member.endswith("tokenizer.model"):
            f = tar.extractfile(member)
            with open(os.path.join(OUTPUT_DIR, "tokenizer.model"), "wb") as out:
                out.write(f.read())
            print("  tokenizer.model extracted")
            break

# ---------------------------------------------------------------------------
# Configure streaming parameters
# ---------------------------------------------------------------------------

print("\nConfiguring streaming parameters...")

if hasattr(model.encoder, "set_default_att_context_size"):
    model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)

streaming_cfg = model.encoder.streaming_cfg
print(f"  Streaming config: {streaming_cfg}")

subsampling_factor = model.cfg.encoder.get("subsampling_factor", 8)
left_context = ATT_CONTEXT_SIZE[0]
right_context = ATT_CONTEXT_SIZE[1]
chunk_size = right_context + 1  # output frames per chunk
print(f"  Subsampling factor : {subsampling_factor}")
print(f"  Left context       : {left_context}")
print(f"  Right context      : {right_context}")
print(f"  Chunk size (frames): {chunk_size}")

model.encoder.setup_streaming_params(chunk_size=chunk_size, shift_size=chunk_size)

drop_extra_pre_encoded = getattr(streaming_cfg, "drop_extra_pre_encoded", 0)
print(f"  drop_extra_pre_encoded: {drop_extra_pre_encoded}")

# ---------------------------------------------------------------------------
# Initial encoder cache state -- I read the shapes back out so the config.json
# I emit later is sourced from the actual model, not my assumptions.
# ---------------------------------------------------------------------------

batch_size = 1
cache_last_channel, cache_last_time, cache_last_channel_len = (
    model.encoder.get_initial_cache_state(batch_size=batch_size)
)

print(f"\nCache dimensions:")
print(f"  cache_last_channel    : {cache_last_channel.shape}")
print(f"  cache_last_time       : {cache_last_time.shape}")
print(f"  cache_last_channel_len: {cache_last_channel_len.shape}")

num_layers = cache_last_channel.shape[0]
hidden_dim = cache_last_channel.shape[3]
conv_context = cache_last_time.shape[3]

# ---------------------------------------------------------------------------
# Sample mel features via NeMo's streaming buffer. I just need one chunk's
# worth of shape-correct input to drive tracing and the verification step.
# ---------------------------------------------------------------------------

print("\nCreating test inputs via NeMo streaming buffer...")

from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=model,
    online_normalization=False,
    pad_and_drop_preencoded=True,
)

# Two seconds of noise is enough for one full chunk.
sample_rate = 16000
dummy_audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1

temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
sf.write(temp_wav.name, dummy_audio, sample_rate)
temp_wav.close()

streaming_buffer.append_audio_file(temp_wav.name, stream_id=-1)
processed_signal, processed_signal_length = next(iter(streaming_buffer))
print(f"  Mel features shape : {processed_signal.shape}")
print(f"  Mel features length: {processed_signal_length}")

os.unlink(temp_wav.name)

# ---------------------------------------------------------------------------
# Pull the prompt dictionary out of the .nemo cfg. This is the table I dump
# to config.json so downstream consumers (the Rust loader) can map a language
# code to a prompt index without re-reading the .nemo.
# ---------------------------------------------------------------------------

prompt_dict = OmegaConf.to_container(
    model.cfg.model_defaults.prompt_dictionary, resolve=True
)
print(f"\nprompt_dictionary: {len(prompt_dict)} entries")

# ---------------------------------------------------------------------------
# Reference inference. I run NeMo's own pipeline (encoder + _apply_prompt_to_encoded)
# end-to-end with the en-US prompt set, and stash the result so I can compare
# against the ONNX output after export.
# ---------------------------------------------------------------------------

print("\nRunning NeMo reference inference (en-US)...")

model.set_inference_prompt("en-US")
prompt_idx_t = torch.tensor([model._inference_prompt_index], dtype=torch.long)

with torch.no_grad():
    enc_raw, encoded_len, _, _, _ = model.encoder.cache_aware_stream_step(
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
        keep_all_outputs=False,
        drop_extra_pre_encoded=drop_extra_pre_encoded,
    )
    enc_ref = model._apply_prompt_to_encoded(enc_raw)

print(f"  Encoder + prompt output shape : {enc_ref.shape}")
print(f"  Encoder output length         : {encoded_len}")

# ---------------------------------------------------------------------------
# Encoder wrapper. This is the only "real" change from the English export
# script: I inline NeMo's _apply_prompt_to_encoded into the traced graph and
# expose prompt_index as an actual ONNX input. The body mirrors the source
# in rnnt_bpe_models_prompt.py exactly so the math is identical -- I just
# replaced `self._inference_prompt_index` (a Python int, frozen at trace time)
# with the new tensor input.
# ---------------------------------------------------------------------------

print("\nExporting encoder (with prompt_index input)...")


class EncoderWithPromptWrapper(torch.nn.Module):
    def __init__(self, model, drop_extra):
        super().__init__()
        self.encoder = model.encoder
        self.prompt_kernel = model.prompt_kernel
        self.num_prompts = model.num_prompts
        self.drop_extra = drop_extra

    def forward(
        self,
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        prompt_index,
    ):
        encoded, enc_len, ch_n, tm_n, ln_n = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )

        # Exactly mirror _apply_prompt_to_encoded. encoder output is (B, D, T),
        # I transpose to (B, T, D), build a [B, T, num_prompts] one-hot at the
        # given index, concat on the last axis, run the kernel, transpose back.
        encoded = encoded.transpose(1, 2)
        B, T, _ = encoded.shape
        prompt = torch.zeros(
            B, T, self.num_prompts, dtype=encoded.dtype, device=encoded.device
        )
        prompt.scatter_(
            2,
            prompt_index.view(B, 1, 1).expand(-1, T, -1),
            1.0,
        )
        encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
        encoded = encoded.transpose(1, 2)
        return encoded, enc_len, ch_n, tm_n, ln_n


wrapper = EncoderWithPromptWrapper(model, drop_extra_pre_encoded).eval()

# Quick parity check before I commit ~2.5 GB of weights to disk: my wrapper
# should produce the same numbers as model._apply_prompt_to_encoded.
with torch.no_grad():
    enc_wrap, _, _, _, _ = wrapper(
        processed_signal, processed_signal_length,
        cache_last_channel, cache_last_time, cache_last_channel_len,
        prompt_idx_t,
    )
diff = (enc_wrap - enc_ref).abs().max().item()
print(f"  Wrapper vs NeMo max diff (pre-export): {diff:.2e}")
if diff > 1e-4:
    raise RuntimeError(
        "Wrapper does not match NeMo. Aborting before writing 2.5 GB of "
        "wrong weights to disk. Re-check _apply_prompt_to_encoded source."
    )

input_names = [
    "processed_signal",
    "processed_signal_length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
    "prompt_index",
]
output_names = [
    "encoded",
    "encoded_len",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_len_next",
]

temp_encoder_path = os.path.join(OUTPUT_DIR, "encoder_temp.onnx")

torch.onnx.export(
    wrapper,
    (
        processed_signal, processed_signal_length,
        cache_last_channel, cache_last_time, cache_last_channel_len,
        prompt_idx_t,
    ),
    temp_encoder_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=17,
    dynamic_axes={
        "processed_signal":        {0: "batch", 2: "time"},
        "processed_signal_length": {0: "batch"},
        "prompt_index":            {0: "batch"},
        "encoded":                 {0: "batch", 2: "time"},
        "encoded_len":             {0: "batch"},
    },
)
print("  Encoder graph exported")

# PyTorch scatters weights across dozens of tiny files by default. I re-save
# everything into a single encoder.onnx + encoder.onnx.data pair so users
# only have to deal with two files. Same trick as in the English-only script.
print("  Consolidating encoder weights into single file...")

encoder_model = onnx.load(temp_encoder_path, load_external_data=True)
final_encoder_path = os.path.join(OUTPUT_DIR, "encoder.onnx")

onnx.save_model(
    encoder_model,
    final_encoder_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="encoder.onnx.data",
    size_threshold=0,
)

del encoder_model
gc.collect()

# Clean up the scattered weight files from the initial export
for pattern in [
    "encoder_temp*", "*.weight", "*MatMul*",
    "Constant_*", "onnx__*", "encoder.pre_encode*",
]:
    for f in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

print("  Encoder saved: encoder.onnx + encoder.onnx.data")

# ---------------------------------------------------------------------------
# Decoder/joint export. There is nothing prompt-specific here -- the RNNT
# decoder and joint network are identical to the English-only model's, just
# wider vocab (~13k vs 1024). I let NeMo's own exporter handle them.
# ---------------------------------------------------------------------------

print("\nExporting decoder/joint...")

temp_decoder_prefix = os.path.join(OUTPUT_DIR, "temp_model")
with torch.no_grad():
    model.export(output=temp_decoder_prefix + ".onnx", check_trace=False)

# NeMo's model.export() dumps both encoder and decoder. I already have my own
# encoder export with the prompt input, so I keep only the decoder_joint here
# and discard the rest.
final_decoder_path = os.path.join(OUTPUT_DIR, "decoder_joint.onnx")
for f in glob.glob(os.path.join(OUTPUT_DIR, "*.onnx")):
    fname = os.path.basename(f).lower()
    if "decoder" in fname and "joint" in fname:
        if f != final_decoder_path:
            shutil.move(f, final_decoder_path)
        break

keep = {"encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model"}
for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
    if os.path.basename(f) not in keep and os.path.isfile(f):
        try:
            os.remove(f)
        except OSError:
            pass

print("  Decoder saved: decoder_joint.onnx")

# ---------------------------------------------------------------------------
# Save config.json. The Rust loader auto-detects the multilingual variant by
# spotting the `prompt_index` ONNX input, so it doesn't strictly need this
# file. I still write it because (a) it's useful for any non-Rust consumer
# and (b) the prompt_dictionary is far less effort to read here than to
# hardcode downstream.
# ---------------------------------------------------------------------------

config = {
    "model_name": "nemotron-3.5-asr-streaming-0.6b",
    "sample_rate": 16000,
    "n_mels": 128,
    "subsampling_factor": subsampling_factor,
    "att_context_size": ATT_CONTEXT_SIZE,
    "left_context": left_context,
    "right_context": right_context,
    "chunk_size_output_frames": chunk_size,
    "drop_extra_pre_encoded": drop_extra_pre_encoded,
    "num_encoder_layers": num_layers,
    "hidden_dim": hidden_dim,
    "conv_context": conv_context,
    "vocab_size": model.tokenizer.vocab_size,
    "blank_id": model.tokenizer.vocab_size,
    "num_prompts": int(model.num_prompts),
    "prompt_dictionary": prompt_dict,
    "preprocessor": {
        "window_size": 0.025,
        "window_stride": 0.01,
        "n_fft": 512,
        "normalize": "NA",
        "preemph": 0.97,
    },
    "cache_shapes": {
        "cache_last_channel": list(cache_last_channel.shape),
        "cache_last_time": list(cache_last_time.shape),
        "cache_last_channel_len": [1],
    },
    "test_input": {
        "mel_shape": list(processed_signal.shape),
        "mel_length": int(processed_signal_length[0]),
        "prompt_index": int(prompt_idx_t[0]),
    },
    "test_output": {
        "encoded_shape": list(enc_ref.shape),
        "encoded_len": int(encoded_len[0]),
    },
}

config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nConfiguration saved to {config_path}")

# ---------------------------------------------------------------------------
# Verify ONNX matches NeMo, and crucially do it for more than one language.
# A single-language verification would not catch a "prompt_index input is
# accepted but ignored" bug -- the output would just look correct because
# en-US happens to be index 0 (all-zero one-hot picks the first prompt row).
# Picking es-ES (idx 2) and ja-JP (idx 10) makes that failure mode impossible.
# ---------------------------------------------------------------------------

print("\nVerifying ONNX exports...")

print("\n  Encoder:")
enc_session = ort.InferenceSession(final_encoder_path, providers=["CPUExecutionProvider"])
for inp in enc_session.get_inputs():
    print(f"    input  {inp.name}: {inp.shape}")
for out in enc_session.get_outputs():
    print(f"    output {out.name}: {out.shape}")

for lang in VERIFY_LANGS:
    if lang not in prompt_dict:
        print(f"    [skip] '{lang}' not in prompt_dictionary")
        continue
    idx = prompt_dict[lang]

    # NeMo reference path
    model.set_inference_prompt(lang)
    with torch.no_grad():
        e_raw, e_len, _, _, _ = model.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )
        e_ref = model._apply_prompt_to_encoded(e_raw).numpy()

    # ONNX path
    e_onnx = enc_session.run(None, {
        "processed_signal":        processed_signal.numpy(),
        "processed_signal_length": processed_signal_length.numpy(),
        "cache_last_channel":      cache_last_channel.numpy(),
        "cache_last_time":         cache_last_time.numpy(),
        "cache_last_channel_len":  cache_last_channel_len.numpy(),
        "prompt_index":            np.array([idx], dtype=np.int64),
    })[0]
    d = np.abs(e_ref - e_onnx).max()
    print(f"    {lang:6s} idx={idx:3d}: max diff = {d:.2e}")

del enc_session

print("\n  Decoder:")
dec_session = ort.InferenceSession(final_decoder_path, providers=["CPUExecutionProvider"])
for inp in dec_session.get_inputs():
    print(f"    input  {inp.name}: {inp.shape}")
for out in dec_session.get_outputs():
    print(f"    output {out.name}: {out.shape}")
del dec_session
gc.collect()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Export complete")
print("=" * 60)

print(f"\nOutput directory: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 ** 2)
    print(f"  {f} ({size_mb:.1f} MB)")

print(f"\nTest: mel {list(processed_signal.shape)} -> encoded {list(enc_ref.shape)}")
print(f"Available languages: {len(prompt_dict)} (see config.json)")
