import torch
import nemo.collections.asr as nemo_asr
import os
import glob
import shutil
import tarfile
import onnxruntime as ort
import logging

# Disable nemo internal type checking.
# This is necessary because the ONNX exporter passes arguments by position,
# but nemos strict typing decorators require keyword arguments.
logging.getLogger('nemo_logging').setLevel(logging.ERROR)
try:
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(False)
except ImportError:
    pass

#cfg

# l_n_p: Local NeMo Path
l_n_p = "parakeet_realtime_eou_120m-v1.nemo"
# m_n: Model Name (HuggingFace ID)
m_n = "nvidia/parakeet_realtime_eou_120m-v1"
# o_d: out dir
o_d = "./onnx_export_streaming"

if os.path.exists(o_d):
    shutil.rmtree(o_d)
os.makedirs(o_d, exist_ok=True)

# o_p: ONNX Prefix
o_p = os.path.join(o_d, "parakeet_eou")

device = torch.device("cpu")
print(f"Loading model on {device}...")

if os.path.exists(l_n_p):
    model = nemo_asr.models.ASRModel.restore_from(l_n_p, map_location=device)
else:
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=m_n, map_location=device)

model.eval()

print("Extracting Tokenizer model...")
# We extract the sentencepiece model directly from the tar archive
# to ensure we have the exact vocab file required for the Rust client.
try:
    with tarfile.open(l_n_p, "r:*") as tar:
        m_name = next((m for m in tar.getnames() if m.endswith("tokenizer.model")), None)
        if m_name:
            f = tar.extractfile(m_name)
            with open(os.path.join(o_d, "tokenizer.model"), "wb") as out:
                out.write(f.read())
            print(f"Tokenizer extracted to {o_d}")
        else:
            print("Warning: tokenizer.model not found in archive.")
except Exception as e:
    print(f"Tokenizer extraction failed: {e}")

print("Exporting Decoder and Joint modules...")
# Standard export works fine for the Decoder/Joint as they are largely
# independent of the streaming state logic.
with torch.no_grad():
    model.export(output=o_p + ".onnx", check_trace=False)

# Export Stateful Encoder 

print("Configuring Encoder for Stateful Streaming...")

# Calculate streaming parameters.
# We need to determine the internal chunk size based on the subsampling factor.
enc_cfg = model.cfg.encoder
sub_factor = enc_cfg.get("subsampling_factor", 4)

c_size_samples = 16
if 'streaming_cfg' in enc_cfg and enc_cfg.streaming_cfg.chunk_size:
    cs = enc_cfg.streaming_cfg.chunk_size
    c_size_samples = cs[1] if isinstance(cs, list) else cs

# s_c_s: Stream Chunk Size (internal)
s_c_s = c_size_samples // sub_factor

model.encoder.setup_streaming_params(
    chunk_size=s_c_s,
    shift_size=s_c_s
)
model.encoder.streaming = True

# Wrapper class to handle ONNX export logic.
# This wrapper serves two purposes:
# 1. It maps positional arguments (from ONNX runtime) to keyword arguments (required by NeMo).
# 2. It calls `cache_aware_stream_step` instead of `forward`. This is the critical change from my inital export (remember my first export)
# that exposes the input/output cache tensors (for stateful streaming)
class NeMoEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Determine drop_extra_pre_encoded behavior safely.
        # We use getattr because streaming_cfg is a DictConfig object, not a standard dict.
        self.drop_extra = True
        if hasattr(encoder, 'streaming_cfg') and encoder.streaming_cfg is not None:
             self.drop_extra = getattr(encoder.streaming_cfg, 'drop_extra_pre_encoded', True)

    def forward(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        # Note: 'audio_signal' here refers to the processed MelSpectrogram features.
        return self.encoder.cache_aware_stream_step(
            processed_signal=audio_signal,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra
        )

# dummy inputs for the export trace.
d_aud = torch.randn(1, 16000)
d_len = torch.tensor([16000])

#  Mel Features (which the encoder expects).
features, feat_lengths = model.preprocessor(input_signal=d_aud, length=d_len)

# initial cache state to determine tensor shapes.
cache_state = model.encoder.get_initial_cache_state(batch_size=1)

wrapper = NeMoEncoderWrapper(model.encoder)
enc_onnx_path = os.path.join(o_d, "encoder.onnx")

# Define input/output names for the ONNX graph.
i_names = ['audio_signal', 'length', 'cache_last_channel', 'cache_last_time', 'cache_last_channel_len']
o_names = ['outputs', 'encoded_lengths', 'new_cache_last_channel', 'new_cache_last_time', 'new_cache_last_channel_len']

print(f"Exporting Stateful Encoder to {enc_onnx_path}...")

torch.onnx.export(
    wrapper,
    (features, feat_lengths, *cache_state),
    enc_onnx_path,
    input_names=i_names,
    output_names=o_names,
    opset_version=17,
    dynamic_axes={
        'audio_signal': {0: 'batch', 2: 'time'},
        'length': {0: 'batch'},
        'outputs': {0: 'batch', 2: 'time'},
        # Cache axes: Batch is dynamic, but feature dims are fixed.
        'cache_last_channel': {0: 'batch_cache'},
        'cache_last_time': {0: 'batch_cache'},
        'cache_last_channel_len': {0: 'batch_cache'}
    }
)

# some organizationS

for f in glob.glob(os.path.join(o_d, "*.onnx")):
    fname = os.path.basename(f).lower()
    
    # Normalize filenames for my Rust side
    if "decoder" in fname and "joint" in fname and "decoder_joint.onnx" not in fname:
         shutil.move(f, os.path.join(o_d, "decoder_joint.onnx"))
    # Remove any stateless encoder exports that might have been generated by the standard export
    elif "encoder" in fname and f != enc_onnx_path:
        os.remove(f)

if os.path.exists(enc_onnx_path):
    sess = ort.InferenceSession(enc_onnx_path, providers=['CPUExecutionProvider'])
    inputs = [i.name for i in sess.get_inputs()]
    
    if "cache_last_channel" in inputs:
        print("Success: Encoder export is stateful (cache inputs detected).")
    else:
        print("Failure: Encoder export appears stateless.")
else:
    print("Error: Encoder ONNX file was not generated.")

print(f"done, look here {o_d}")
