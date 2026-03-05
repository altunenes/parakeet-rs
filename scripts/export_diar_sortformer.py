import argparse

parser = argparse.ArgumentParser(description = "Export model to ONNX with custom streaming parameters")
parser.add_argument("input_path", help = "Path to input model (eg diar_streaming_sortformer_4spk-v2.1.nemo)")
parser.add_argument("output_path", help = "Path to onnx export (eg diar_streaming_sortformer_4spk-v2.1.onnx)")
parser.add_argument("--chunk-len", help = "The number of frames in a processing chunk", type=int, default=124)
parser.add_argument("--right-context", help = "The number of future frames attached after the chunk.", type=int, default=1)
parser.add_argument("--fifo-len", help = "The number of previous frames attached before the chunk, from the FIFO queue.", type=int, default=124)
parser.add_argument("--spkcache-update-period", help = "The number of frames extracted from the FIFO queue to update the speaker cache.", type=int, default=144)
parser.add_argument("--spkcache-len", help = "The total number of frames in the speaker cache.", type=int, default=188)

args = parser.parse_args()

import torch
import types
from nemo.collections.asr.models import SortformerEncLabelModel

print(f"PyTorch version: {torch.__version__}")

print(f"loading model from: {args.input_path}")
model = SortformerEncLabelModel.restore_from(restore_path=args.input_path, map_location='cpu')
model.eval()

FEAT_DIM = 128
EMB_DIM = 512

model.sortformer_modules.chunk_len = args.chunk_len
model.sortformer_modules.chunk_right_context = args.right_context
model.sortformer_modules.fifo_len = args.fifo_len
model.sortformer_modules.spkcache_update_period = args.spkcache_update_period
model.sortformer_modules.spkcache_len = args.spkcache_len
model.sortformer_modules._check_streaming_parameters()

def onnx_forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
    chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
    chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

    concat_embs = torch.cat([spkcache, fifo, chunk_pre_encode_embs], dim=1)
    concat_lens = spkcache_lengths + fifo_lengths + chunk_pre_encode_lengths

    spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
        processed_signal=concat_embs,
        processed_signal_length=concat_lens,
        bypass_pre_encode=True,
    )

    spkcache_fifo_chunk_preds = self.forward_infer(
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
    )

    return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

model.forward = types.MethodType(onnx_forward, model)

batch_size = 1
subsampling = 8
chunk_frames_in = args.chunk_len * subsampling

chunk = torch.randn(batch_size, chunk_frames_in, FEAT_DIM)
chunk_lengths = torch.tensor([chunk_frames_in], dtype=torch.long)
spkcache = torch.zeros(batch_size, args.spkcache_len, EMB_DIM)
spkcache_lengths = torch.tensor([0], dtype=torch.long)
fifo = torch.zeros(batch_size, args.fifo_len, EMB_DIM)
fifo_lengths = torch.tensor([0], dtype=torch.long)

input_example = (chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths)

print(f"  chunk    {chunk.shape}")
print(f"  spkcache: {spkcache.shape}")
print(f"  fif:     {fifo.shape}")

torch.onnx.export(
    model,
    input_example,
    args.output_path,
    input_names=["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
    output_names=["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
    dynamic_axes={
        "chunk": {0: "batch", 1: "time_chunk"},
        "spkcache": {0: "batch", 1: "time_cache"},
        "fifo": {0: "batch", 1: "time_fifo"},
        "spkcache_fifo_chunk_preds": {0: "batch", 1: "time_out"},
        "chunk_pre_encode_embs": {0: "batch", 1: "time_pre_encode"}
    },
    opset_version=17,
    dynamo=False,
    verbose=False,
)

print(f"xported to: {args.output_path}")


import onnx
model_onnx = onnx.load(args.output_path)
print("\n verify input shapes:")
for inp in model_onnx.graph.input:
    dims = [d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")

fifo_input = [inp for inp in model_onnx.graph.input if inp.name == "fifo"][0]
fifo_dim1 = fifo_input.type.tensor_type.shape.dim[1]
if fifo_dim1.dim_param:
    print(f"\n FIFO is dynamic: '{fifo_dim1.dim_param}'")
else:
    print(f"\n FIFO is fixed: {fifo_dim1.dim_value}")

model_onnx.metadata_props.append(onnx.StringStringEntryProto(key="chunk_len", value=str(args.chunk_len)))
model_onnx.metadata_props.append(onnx.StringStringEntryProto(key="fifo_len", value=str(args.fifo_len)))
model_onnx.metadata_props.append(onnx.StringStringEntryProto(key="spkcache_len", value=str(args.spkcache_len)))
model_onnx.metadata_props.append(onnx.StringStringEntryProto(key="right_context", value=str(args.right_context)))

print("\nSaving model with custom metadata")

onnx.save(model_onnx, args.output_path)
