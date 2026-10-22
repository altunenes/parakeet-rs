# Export multitalker-parakeet-streaming-0.6b-v1 to ONNX with optional int8 quantisation.
#
# Original source and pre-exported ONNX files:
#   https://huggingface.co/smcleod/multitalker-parakeet-streaming-0.6b-v1-onnx-int8
#
# This model differs from the standard Nemotron streaming model: the encoder
# accepts speaker target masks (spk_targets, bg_spk_targets) as additional inputs.
# These masks enable speaker kernel injection at configured encoder layers.
#
# The standard NeMo .export() does NOT include speaker targets as ONNX inputs
# because speaker kernels are applied via forward hooks that read instance
# attributes, which become constants during ONNX tracing. We work around this
# with a custom wrapper that sets speaker targets on the model before calling
# encoder.forward_for_export(), keeping the hooks active so the exporter follows
# the data flow through the kernel FFNs.
#
# Graphs are exported with PyTorch's dynamo exporter (opset 23); onnx_export.py in this
# folder holds the shared settings, keep it next to this script.
#
# Requirements:
#   pip install nemo_toolkit[asr] onnx onnxscript onnxruntime
#
# Usage:
#   python export_multitalker.py --nemo-path /path/to/model.nemo --output-dir ./out
#   python export_multitalker.py --nemo-path /path/to/model.nemo --output-dir ./out --no-quantise
#
# Output:
#   encoder.onnx + encoder.onnx.data   (7 inputs: signal, length, cache x3, spk/bg targets)
#   decoder_joint.onnx                 (standard RNNT)
#   tokenizer.model                    (SentencePiece)
#   multitalker_config.json            (model dimensions for Rust inference)
#   encoder.int8.onnx, decoder_joint.int8.onnx (unless --no-quantise)

import argparse
import json
import os
import sys
import tarfile
import tempfile

import onnx
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr

import onnx_export


class MultitalkerEncoderExport(nn.Module):
    """Wrapper that exposes spk_targets and bg_spk_targets as explicit forward() inputs.

    The original encoder uses forward hooks registered by the model to inject speaker
    kernels. The hooks read spk_targets/bg_spk_targets from the model instance and use
    spk_kernels/bg_spk_kernels (also on the model). This wrapper:
    1. Keeps hooks active on encoder layers
    2. Owns the kernel submodules (so their params appear in the ONNX graph)
    3. Sets the traced tensor on the model before calling encoder.forward_for_export()
    """

    def __init__(self, asr_model):
        super().__init__()
        self.encoder = asr_model.encoder
        # Own the kernel modules so their parameters are included in the ONNX graph
        self.spk_kernels = asr_model.spk_kernels
        self.bg_spk_kernels = asr_model.bg_spk_kernels
        # Store model reference in a list so it's not registered as a submodule
        self._model_ref = [asr_model]

    def forward(
        self,
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        spk_targets,
        bg_spk_targets,
    ):
        model = self._model_ref[0]
        model.spk_targets = spk_targets
        model.bg_spk_targets = bg_spk_targets

        return self.encoder.forward_for_export(
            audio_signal=processed_signal,
            length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )


class DecoderJointExport(nn.Module):
    """One RNNT prediction step: decoder LSTM + joint, called with keyword arguments
    (NeMo's typed methods require them). encoder_outputs is [batch, time, d_model]."""

    def __init__(self, decoder, joint):
        super().__init__()
        self.decoder = decoder
        self.joint_enc = joint.enc
        self.joint_pred = joint.pred
        self.joint_net = joint.joint_net

    def forward(self, encoder_outputs, targets, input_states_1, input_states_2):
        dec_out, dec_states = self.decoder.predict(
            y=targets, state=[input_states_1, input_states_2], add_sos=False
        )
        enc_proj = self.joint_enc(encoder_outputs)
        pred_proj = self.joint_pred(dec_out)
        combined = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
        length = torch.tensor(dec_out.shape[1], dtype=torch.int64)
        return self.joint_net(combined), length, dec_states[0], dec_states[1]


def extract_tokenizer(nemo_path, output_dir):
    """Extract tokenizer.model from the .nemo archive."""
    with tarfile.open(nemo_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.endswith("tokenizer.model"):
                with open(os.path.join(output_dir, "tokenizer.model"), "wb") as out:
                    out.write(tar.extractfile(member).read())
                print(f"  Extracted tokenizer: {member.name}")
                return
    raise FileNotFoundError(f"Could not find tokenizer.model in {nemo_path}")


def get_model_config(asr_model):
    """Extract model dimensions and streaming config from the loaded NeMo model."""
    enc_cfg = asr_model.cfg.encoder
    dec = asr_model.decoder
    enc = asr_model.encoder
    scfg = enc.streaming_cfg

    d_model = enc_cfg.d_model
    n_layers = enc_cfg.get("n_layers", None) or len(enc.layers)
    subsampling_factor = enc_cfg.get("subsampling_factor", 8)

    left_context = scfg.last_channel_cache_size

    # Conv cache = depthwise conv kernel_size - 1
    dw_conv = enc.layers[0].conv.depthwise_conv
    kernel_size = dw_conv.kernel_size[0] if not hasattr(dw_conv, "conv") else dw_conv.conv.kernel_size[0]
    conv_context = kernel_size - 1

    # Streaming chunk config
    chunk_sizes = scfg.chunk_size
    pre_encode_cache = scfg.pre_encode_cache_size
    chunk_size = chunk_sizes[1] if isinstance(chunk_sizes, (list, tuple)) and len(chunk_sizes) > 1 else chunk_sizes
    pre_enc = pre_encode_cache[1] if isinstance(pre_encode_cache, (list, tuple)) and len(pre_encode_cache) > 1 else pre_encode_cache
    drop_extra_pre_encoded = scfg.drop_extra_pre_encoded

    # Speaker kernel layers
    spk_kernel_layers = []
    if hasattr(asr_model.cfg, "spk_kernel_layers"):
        spk_kernel_layers = list(asr_model.cfg.spk_kernel_layers)
    elif hasattr(asr_model, "spk_kernel_layers"):
        spk_kernel_layers = list(asr_model.spk_kernel_layers)

    att_context = list(enc_cfg.get("att_context_size", [[70, 13]]))

    return {
        "model_type": "multitalker-parakeet-streaming",
        "encoder": {
            "d_model": d_model,
            "num_layers": n_layers,
            "subsampling_factor": subsampling_factor,
            "left_context": left_context,
            "conv_context": conv_context,
            "chunk_size": chunk_size,
            "pre_encode_cache_size": pre_enc,
            "drop_extra_pre_encoded": drop_extra_pre_encoded,
            "att_context_size": att_context,
        },
        "decoder": {
            "vocab_size": dec.vocab_size,
            "pred_hidden": dec.pred_hidden,
            "pred_rnn_layers": dec.pred_rnn_layers,
            "blank_id": dec.vocab_size,
        },
        "speaker_kernels": {
            "spk_kernel_layers": spk_kernel_layers,
            "spk_kernel_type": getattr(asr_model, "spk_kernel_type", "ff"),
            "add_bg_spk_kernel": getattr(asr_model, "add_bg_spk_kernel", True),
        },
        "preprocessor": {
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 512,
            "normalize": "NA",
        },
    }


def export_encoder(asr_model, output_path, config, metadata):
    """Export the encoder with explicit speaker target inputs via the hook-based wrapper."""
    wrapper = MultitalkerEncoderExport(asr_model)

    enc = config["encoder"]
    time_steps = enc["chunk_size"] + enc["pre_encode_cache_size"]

    # Cache format is [batch, n_layers, ...] (forward_for_export transposes internally)
    inputs = (
        torch.randn(1, 128, time_steps),
        torch.tensor([time_steps], dtype=torch.long),
        torch.zeros(1, enc["num_layers"], enc["left_context"], enc["d_model"]),
        torch.zeros(1, enc["num_layers"], enc["d_model"], enc["conv_context"]),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, time_steps),
        torch.zeros(1, time_steps),
    )
    print(f"  Signal shape: [1, 128, {time_steps}]")
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "encoder.onnx")
        onnx_export.export(
            wrapper,
            inputs,
            raw,
            input_names=[
                "processed_signal", "processed_signal_length",
                "cache_last_channel", "cache_last_time", "cache_last_channel_len",
                "spk_targets", "bg_spk_targets",
            ],
            output_names=[
                "encoded", "encoded_len",
                "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next",
            ],
            dynamic_axes={
                "processed_signal": {2: "time"},
                "spk_targets": {1: "spk_time"},
                "bg_spk_targets": {1: "spk_time"},
            },
        )
        model = onnx.load(raw)
    if "spk_targets" not in [i.name for i in model.graph.input]:
        raise RuntimeError("spk_targets was folded into a constant; the speaker kernels would ignore it")
    onnx_export.save(model, output_path, os.path.basename(output_path) + ".data", metadata)


def export_decoder_joint(asr_model, output_path, config, metadata):
    dec = asr_model.decoder
    inputs = (
        torch.randn(1, 1, config["encoder"]["d_model"]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(dec.pred_rnn_layers, 1, dec.pred_hidden),
        torch.zeros(dec.pred_rnn_layers, 1, dec.pred_hidden),
    )
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "decoder_joint.onnx")
        onnx_export.export(
            DecoderJointExport(dec, asr_model.joint),
            inputs,
            raw,
            input_names=["encoder_outputs", "targets", "input_states_1", "input_states_2"],
            output_names=["outputs", "prednet_lengths", "states_1", "states_2"],
            dynamic_axes={"encoder_outputs": {1: "enc_time"}},
        )
        onnx_export.save(onnx_export.fold_constants(onnx.load(raw)), output_path, metadata=metadata)


def quantise(output_dir, metadata):
    """Dynamic int8 quantisation of both graphs."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name, weight_type in (("encoder", QuantType.QUInt8), ("decoder_joint", QuantType.QInt8)):
        print(f"Quantising {name} to int8 (dynamic)...")
        out = os.path.join(output_dir, f"{name}.int8.onnx")
        quantize_dynamic(
            os.path.join(output_dir, f"{name}.onnx"), out, weight_type=weight_type, op_types_to_quantize=onnx_export.INT8_OPS
        )
        model = onnx.load(out)
        del model.metadata_props[:]
        onnx_export.save(model, out, metadata=metadata)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Export multitalker Parakeet streaming model to ONNX")
    parser.add_argument("--nemo-path", required=True, help="Path to .nemo model file")
    parser.add_argument("--no-quantise", action="store_true", help="Export fp32 models only, skip quantisation")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    if not os.path.exists(args.nemo_path):
        print(f"Model not found: {args.nemo_path}")
        print("Download from: https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.nemo_path}...")
    asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_path, map_location="cpu")
    asr_model.eval()
    onnx_export.use_sdpa(asr_model.encoder)
    onnx_export.exact_position_slices(asr_model.encoder)

    config = get_model_config(asr_model)
    with open(os.path.join(args.output_dir, "multitalker_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\nExtracting tokenizer...")
    extract_tokenizer(args.nemo_path, args.output_dir)

    dec = asr_model.decoder
    metadata = {
        "vocab_size": dec.vocab_size,
        "normalize_type": "",
        "pred_rnn_layers": dec.pred_rnn_layers,
        "pred_hidden": dec.pred_hidden,
        "subsampling_factor": config["encoder"]["subsampling_factor"],
        "model_type": "MultitalkerEncDecRNNTBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1",
        "feat_dim": 128,
        "spk_kernel_layers": str(config["speaker_kernels"]["spk_kernel_layers"]),
    }

    print("\nExporting encoder...")
    export_encoder(asr_model, os.path.join(args.output_dir, "encoder.onnx"), config, metadata)
    print("\nExporting decoder_joint...")
    export_decoder_joint(asr_model, os.path.join(args.output_dir, "decoder_joint.onnx"), config, metadata)

    if not args.no_quantise:
        quantise(args.output_dir, metadata)

    print("\nOutput files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {f}: {os.path.getsize(os.path.join(args.output_dir, f)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
