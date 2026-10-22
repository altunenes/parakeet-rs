# ONNX export settings shared by the export scripts in this folder.
#
# Every graph is exported with PyTorch's dynamo exporter (torch.export + onnxscript) at opset 23,
# which writes attention as the ONNX Attention op and keeps LSTMs as the ONNX LSTM op.
#
# Requirements: torch >= 2.9, onnxscript, onnx.

import os
import tempfile
import types

import onnx
import torch

OPSET = 23

# Op types for ONNX Runtime's dynamic int8 quantization. Its default list includes Attention, which
# it then turns into com.microsoft QAttention with attributes that op does not have; attention
# stays fp32.
INT8_OPS = ["MatMul", "Gemm", "Gather", "LSTM"]


def use_sdpa(module):
    """Route NeMo attention through F.scaled_dot_product_attention, so each layer exports as one
    ONNX Attention node (the relative-position term goes in as an additive mask)."""
    count = 0
    for m in module.modules():
        if hasattr(m, "use_pytorch_sdpa"):
            m.use_pytorch_sdpa = True
            m.use_pytorch_sdpa_backends = None
            count += 1
    print(f"  SDPA attention on {count} modules")
    return count


def _rel_pos_forward(self, x, cache_len=0):
    if self.xscale:
        x = x * self.xscale
    input_len = x.size(1) + cache_len
    center_pos = self.pe.size(1) // 2 + 1
    pos_emb = self.pe.narrow(1, center_pos - input_len, 2 * input_len - 1)
    if self.dropout_emb:
        pos_emb = self.dropout_emb(pos_emb)
    return self.dropout(x), pos_emb


def exact_position_slices(module):
    """NeMo's RelPositionalEncoding slices its table as pe[:, start:end]; with a symbolic input
    length torch.export cannot size that slice. narrow() takes the same 2 * L - 1 rows with an
    exact length."""
    count = 0
    for m in module.modules():
        if type(m).__name__ == "RelPositionalEncoding":
            m.forward = types.MethodType(_rel_pos_forward, m)
            count += 1
    return count


def export(module, inputs, path, input_names, output_names, dynamic_axes=None):
    """Export `module` with the dynamo exporter.

    `dynamic_axes` maps input names to their dynamic axes: a list of axes, or a dict from axis to
    an upper bound (an int) or a name (ignored). Some NeMo code only exports with a bound, e.g. a
    streaming encoder slicing its positional table. Output sizes follow from the inputs.
    """
    bounds = {}
    for name, axes in (dynamic_axes or {}).items():
        items = axes.items() if isinstance(axes, dict) else ((a, None) for a in axes)
        bounds[name] = {int(a): (v if isinstance(v, int) else None) for a, v in items}
    dynamic_shapes = tuple(
        {a: torch.export.Dim.AUTO(max=m) if m else torch.export.Dim.AUTO for a, m in bounds[name].items()}
        if name in bounds else None
        for name in input_names
    )
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(inputs),
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=OPSET,
            dynamo=True,
        )
    # Dim.AUTO lets torch.export fix a size it cannot keep dynamic; refuse that.
    graph_inputs = {v.name: v for v in onnx.load(path, load_external_data=False).graph.input}
    for name, axes in bounds.items():
        dims = graph_inputs[name].type.tensor_type.shape.dim
        fixed = [a for a in axes if dims[a].HasField("dim_value")]
        if fixed:
            raise RuntimeError(f"{name}: axes {fixed} became fixed sizes in the export")


def export_decoder_joint(model, path):
    """Export NeMo's RNNT/TDT decoder + joint (one prediction step) as one .onnx file, with NeMo's
    own example inputs, input/output names and dynamic axes."""
    from nemo.core.classes.common import typecheck
    from nemo.core.classes.exportable import Exportable

    decoder_joint = model.decoder_joint
    decoder_joint.eval()
    typecheck.set_typecheck_enabled(enabled=False)
    for m in decoder_joint.modules():
        if isinstance(m, Exportable) and m is not decoder_joint:
            m._prepare_for_export(noreplace=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "decoder_joint.onnx")
        export(
            decoder_joint,
            decoder_joint.input_example(),
            raw,
            input_names=decoder_joint.input_names,
            output_names=decoder_joint.output_names,
            dynamic_axes=decoder_joint.dynamic_shapes_for_export(),
        )
        save(fold_constants(onnx.load(raw)), path)


def fold_constants(model):
    """Fold every constant subgraph of a small graph into initializers. The dynamo exporter
    builds an LSTM's W and R from the PyTorch weights at runtime (gate reordering); folded,
    they are plain weights like in any other ONNX LSTM."""
    import onnx_ir as ir
    from onnxscript import optimizer

    ir_model = ir.serde.deserialize_model(model)
    optimizer.fold_constants(ir_model, input_size_limit=2**31, output_size_limit=2**31)
    ir.passes.common.RemoveUnusedNodesPass()(ir_model)
    return ir.serde.serialize_model(ir_model)


def save(model, path, data_file=None, metadata=None):
    """Save `model` with its weights in one external data file (or none if `data_file` is None),
    adding `metadata` as metadata_props.

    Tensors under 1 KB stay in the .onnx file: ONNX Runtime cannot read the dynamo export's shape
    constants from external data.
    """
    for key, value in (metadata or {}).items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=str(key), value=str(value)))
    if data_file is None:
        onnx.save_model(model, path)
    else:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
