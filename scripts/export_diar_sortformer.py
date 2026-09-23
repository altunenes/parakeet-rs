# Sortformer v3, to ONNX for parakeet-rs.
#
# Requires a recent NVIDIA NeMo Speech install and the .nemo checkpoint:
#   https://huggingface.co/nvidia/Nemotron-3-Diarization
#
# About this export:
#
# 1. FlexAttention. v3's encoder builds its attention mask with FlexAttention
#    (create_block_mask/flex_attention), which torch.onnx.export cannot trace. We swap
#    it for scaled_dot_product_attention during export only. This is exact for this model
#    (RoPE is applied to q/k directly, attn_mode is full, so the mask is a plain padding
#    mask). We can drop the shim once NeMo ships an export friendly attention path.
#
# 2. Dual res output. The model predicts at 10ms and NeMo averages that down to 80ms
#    frames. We export both: `preds_diar` (80ms) updates the FIFO and speaker cache, exactly as
#    NeMo's streaming does, and `preds_hires` (10ms) is what parakeet-rs returns. The
#    downsampling is an average pool, so the second output costs nothing extra.
#
# Streaming params are CLI args and are also written to metadata as the runtime's default
# profile. The graph uses dynamic axes, so one export serves every latency preset in NVIDIA's
# model card; parakeet-rs switches between them at runtime (StreamingProfile).
#


import argparse

parser = argparse.ArgumentParser(description="Export Nemotron-3 Diarization to ONNX (dual-resolution).")
parser.add_argument("input_path", help="Path to input model (eg Nemotron-3-Diarization.nemo)")
parser.add_argument("output_path", help="Path to onnx export (eg nemotron3_diar_v3.onnx)")
parser.add_argument("--chunk-len", type=int, default=340, help="Frames in a processing chunk.")
parser.add_argument("--right-context", type=int, default=40, help="Future frames attached after the chunk.")
parser.add_argument("--fifo-len", type=int, default=40, help="Frames attached before the chunk from the FIFO queue.")
parser.add_argument("--spkcache-update-period", type=int, default=300, help="Frames popped from FIFO to update the speaker cache.")
parser.add_argument("--spkcache-len", type=int, default=264, help="Total frames in the speaker cache.")
args = parser.parse_args()

import sys
import types
import torch
import torch.nn.functional as F
from nemo.collections.asr.models import SortformerEncLabelModel

print(f"PyTorch version: {torch.__version__}")
print(f"loading model from: {args.input_path}")
model = SortformerEncLabelModel.restore_from(restore_path=args.input_path, map_location="cpu", strict=False)
model.eval()

model.sortformer_modules.chunk_len = args.chunk_len
model.sortformer_modules.chunk_right_context = args.right_context
model.sortformer_modules.fifo_len = args.fifo_len
model.sortformer_modules.spkcache_update_period = args.spkcache_update_period
model.sortformer_modules.spkcache_len = args.spkcache_len
model._check_streaming_parameters()

# --- FlexAttention -> scaled_dot_product_attention shim (export only) ---
import torch.nn.attention.flex_attention as flex_mod

_orig_create_block_mask = flex_mod.create_block_mask
_orig_flex_attention = flex_mod.flex_attention

class _DenseMask:
    def __init__(self, dense):
        self.dense = dense

def _create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None, device="cpu", **_):
    b = torch.arange(B or 1, device=device).view(-1, 1, 1, 1)
    h = torch.arange(H or 1, device=device).view(1, -1, 1, 1)
    q = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    k = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)
    return _DenseMask(mask_mod(b, h, q, k))  # dense boolean mask, no vmap -> traceable

def _flex_attention(query, key, value, score_mod=None, block_mask=None, scale=None, **_):
    assert score_mod is None, "score_mod is unsupported by the SDPA export shim"
    mask = block_mask.dense if isinstance(block_mask, _DenseMask) else block_mask
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)

for _mod in list(sys.modules.values()):
    if getattr(_mod, "create_block_mask", None) is _orig_create_block_mask:
        _mod.create_block_mask = _create_block_mask
    if getattr(_mod, "flex_attention", None) is _orig_flex_attention:
        _mod.flex_attention = _flex_attention
flex_mod.create_block_mask = _create_block_mask
flex_mod.flex_attention = _flex_attention

# Export forward: NVIDIA's forward_for_export, returning both resolutions
def onnx_forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
    chunk_pre_encode_embs, chunk_pre_encode_lengths = self._call_pre_encode(chunk, chunk_lengths)
    chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

    output_length = spkcache.shape[1] + fifo.shape[1] + chunk_pre_encode_embs.shape[1]
    concat_embs, concat_lengths = self.sortformer_modules.concat_and_pad(
        [spkcache, fifo, chunk_pre_encode_embs],
        [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths],
        output_length=output_length,
    )

    encoder_embs, encoder_lengths = self.frontend_encoder(
        processed_signal=concat_embs,
        processed_signal_length=concat_lengths,
        bypass_pre_encode=True,
    )

    preds_hires = self.forward_infer(encoder_embs, encoder_lengths)
    if self.high_resolution:
        preds_diar = self.sortformer_modules.downsample_preds(preds_hires, self.upsample_factor)
    else:
        preds_diar = preds_hires

    return preds_diar, preds_hires, chunk_pre_encode_embs, chunk_pre_encode_lengths

model.forward = types.MethodType(onnx_forward, model)

batch_size = 1
subsampling = model.encoder.subsampling_factor
feat_dim = model.encoder._feat_in
emb_dim = model.sortformer_modules.fc_d_model
chunk_frames = (model.sortformer_modules.chunk_left_context + args.chunk_len + args.right_context) * subsampling

chunk = torch.randn(batch_size, chunk_frames, feat_dim)
chunk_lengths = torch.tensor([chunk_frames], dtype=torch.long)
spkcache = torch.randn(batch_size, args.spkcache_len, emb_dim)
spkcache_lengths = torch.tensor([args.spkcache_len // 2], dtype=torch.long)
fifo = torch.randn(batch_size, args.fifo_len, emb_dim)
fifo_lengths = torch.tensor([args.fifo_len // 2], dtype=torch.long)
input_example = (chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths)

print(f"  chunk:    {tuple(chunk.shape)}")
print(f"  spkcache: {tuple(spkcache.shape)}")
print(f"  fifo:     {tuple(fifo.shape)}")

torch.onnx.export(
    model,
    input_example,
    args.output_path,
    input_names=["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
    output_names=["preds_diar", "preds_hires", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
    dynamic_axes={
        "chunk": {0: "batch", 1: "time_chunk"},
        "spkcache": {0: "batch", 1: "time_cache"},
        "fifo": {0: "batch", 1: "time_fifo"},
        "preds_diar": {0: "batch", 1: "time_out_diar"},
        "preds_hires": {0: "batch", 1: "time_out_hires"},
        "chunk_pre_encode_embs": {0: "batch", 1: "time_pre_encode"},
    },
    opset_version=17,
    dynamo=False,
    verbose=False,
)
print(f"exported to: {args.output_path}")

import onnx

model_onnx = onnx.load(args.output_path)
print("\nverify input/output shapes:")
for value in list(model_onnx.graph.input) + list(model_onnx.graph.output):
    dims = [d.dim_param if d.dim_param else d.dim_value for d in value.type.tensor_type.shape.dim]
    print(f"  {value.name}: {dims}")

metadata = {
    "chunk_len": args.chunk_len,
    "right_context": args.right_context,
    "fifo_len": args.fifo_len,
    "spkcache_update_period": args.spkcache_update_period,
    "spkcache_len": args.spkcache_len,
    "subsampling_factor": subsampling,
    "upsample_factor": model.sortformer_modules.upsample_factor,
    "n_spk": model.sortformer_modules.n_spk,
    "feat_dim": feat_dim,
    "emb_dim": emb_dim,
    "high_resolution": int(bool(model.high_resolution)),
    "spkcache_sil_frames_per_spk": model.sortformer_modules.spkcache_sil_frames_per_spk,
    "use_learnable_sil_emb": int(bool(model.sortformer_modules.use_learnable_sil_emb)),
}
for key, value in metadata.items():
    model_onnx.metadata_props.append(onnx.StringStringEntryProto(key=str(key), value=str(value)))

# The speaker-cache compression uses a learned silence embedding for the padded cache
# slots; it lives outside the graph, so embed it in metadata (the runtime reads it).
if model.sortformer_modules.use_learnable_sil_emb:
    sil_emb = model.sortformer_modules.learnable_sil_emb.detach().cpu().numpy().reshape(-1)
    model_onnx.metadata_props.append(
        onnx.StringStringEntryProto(
            key="learnable_sil_emb", value=",".join(f"{v:.9g}" for v in sil_emb)
        )
    )

print("\nsaving model with metadata:", metadata)
onnx.save(model_onnx, args.output_path)
