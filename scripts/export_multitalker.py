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
# encoder.forward_for_export(), keeping the hooks active so the tracer follows
# the data flow through the kernel FFNs.
#
# Requirements (Python 3.12, PyTorch < 2.9):
#   pip install nemo_toolkit[asr] onnx onnxruntime
#
# Usage:
#   python export_multitalker.py
#   python export_multitalker.py --no-quantise
#   python export_multitalker.py --nemo-path /path/to/model.nemo --output-dir ./out
#
# Output:
#   encoder.onnx + encoder.onnx.data   (7 inputs: signal, length, cache x3, spk/bg targets)
#   decoder_joint.onnx                 (standard RNNT)
#   tokenizer.model                    (SentencePiece)
#   multitalker_config.json            (model dimensions for Rust inference)

import argparse
import json
import os
import sys
import tarfile
import traceback
import zipfile

import onnx
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr


DEFAULT_NEMO_PATH = "../multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo"


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


def extract_tokenizer(nemo_path, output_dir):
    """Extract tokenizer.model from .nemo archive (tar or zip)."""
    tokenizer_out = os.path.join(output_dir, "tokenizer.model")

    try:
        with tarfile.open(nemo_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("tokenizer.model"):
                    f = tar.extractfile(member)
                    if f:
                        with open(tokenizer_out, "wb") as out:
                            out.write(f.read())
                        print(f"  Extracted tokenizer from tar: {member.name}")
                        return tokenizer_out
    except tarfile.TarError:
        pass

    try:
        with zipfile.ZipFile(nemo_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("tokenizer.model"):
                    with zf.open(name) as f, open(tokenizer_out, "wb") as out:
                        out.write(f.read())
                    print(f"  Extracted tokenizer from zip: {name}")
                    return tokenizer_out
    except zipfile.BadZipFile:
        pass

    raise FileNotFoundError(
        f"Could not find tokenizer.model in {nemo_path}. "
        "Try extracting manually: tar xf model.nemo && find . -name tokenizer.model"
    )


def add_meta_data(filename, meta_data, use_external_data=False):
    """Add metadata key-value pairs to an ONNX model."""
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if use_external_data or ("encoder" in filename and "int8" not in filename):
        data_file = filename + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(filename) + ".data",
        )
    else:
        onnx.save(model, filename)


def dynamic_quantise(output_dir, meta_data):
    """Apply dynamic int8 quantisation to exported models."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    encoder_in = os.path.join(output_dir, "encoder.onnx")
    encoder_out = os.path.join(output_dir, "encoder.int8.onnx")
    decoder_in = os.path.join(output_dir, "decoder_joint.onnx")
    decoder_out = os.path.join(output_dir, "decoder_joint.int8.onnx")

    print("Quantising encoder to int8 (dynamic)...")
    quantize_dynamic(encoder_in, encoder_out, weight_type=QuantType.QUInt8)

    print("Quantising decoder_joint to int8 (dynamic)...")
    quantize_dynamic(decoder_in, decoder_out, weight_type=QuantType.QInt8)

    print("Adding metadata...")
    add_meta_data(encoder_out, meta_data)
    add_meta_data(decoder_out, meta_data)

    return encoder_out, decoder_out


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


def export_decoder_joint(asr_model, output_path, config):
    """Export the decoder + joint network.

    NeMo's RNNTDecoderJoint.forward uses positional args which conflicts with
    NeMo's typed-methods decorator requiring kwargs. We build our own wrapper
    that calls decoder.predict() and joint with kwargs to avoid the issue.
    """
    d_model = config["encoder"]["d_model"]
    dec = asr_model.decoder
    joint = asr_model.joint
    pred_hidden = dec.pred_hidden
    pred_rnn_layers = dec.pred_rnn_layers

    class DecoderJointExport(nn.Module):
        def __init__(self, decoder, joint_net):
            super().__init__()
            self.decoder = decoder
            self.joint_enc = joint_net.enc
            self.joint_pred = joint_net.pred
            self.joint_net = joint_net.joint_net

        def forward(self, encoder_outputs, targets, input_states_1, input_states_2):
            dec_out, dec_states = self.decoder.predict(
                y=targets, state=[input_states_1, input_states_2], add_sos=False
            )
            enc_proj = self.joint_enc(encoder_outputs)
            pred_proj = self.joint_pred(dec_out)
            combined = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
            joint_out = self.joint_net(combined)
            return joint_out, dec_out.shape[1], dec_states[0], dec_states[1]

    wrapper = DecoderJointExport(dec, joint)
    wrapper.eval()

    batch = 1
    dummy_encoder = torch.randn(batch, 1, d_model)
    dummy_targets = torch.zeros(batch, 1, dtype=torch.long)
    dummy_state1 = torch.zeros(pred_rnn_layers, batch, pred_hidden)
    dummy_state2 = torch.zeros(pred_rnn_layers, batch, pred_hidden)

    with torch.no_grad():
        test_out = wrapper(dummy_encoder, dummy_targets, dummy_state1, dummy_state2)
        print(f"  Test outputs: {len(test_out)}")
        for i, o in enumerate(test_out):
            if isinstance(o, torch.Tensor):
                print(f"    output[{i}]: {o.shape}")
            else:
                print(f"    output[{i}]: {o}")

    print("  Exporting decoder_joint...")
    torch.onnx.export(
        wrapper,
        (dummy_encoder, dummy_targets, dummy_state1, dummy_state2),
        output_path,
        input_names=["encoder_outputs", "targets", "input_states_1", "input_states_2"],
        output_names=["outputs", "prednet_lengths", "states_1", "states_2"],
        dynamic_axes={"encoder_outputs": {1: "enc_time"}},
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  decoder_joint.onnx: {size_mb:.1f}MB")


def custom_export_encoder(asr_model, output_path, config):
    """Export encoder with explicit speaker target inputs via the hook-based wrapper.

    This is the primary export path. It keeps hooks active on encoder layers and
    sets the traced spk_targets on the model before calling forward_for_export().
    """
    wrapper = MultitalkerEncoderExport(asr_model)
    wrapper.eval()

    d_model = config["encoder"]["d_model"]
    n_layers = config["encoder"]["num_layers"]
    left_context = config["encoder"]["left_context"]
    conv_context = config["encoder"]["conv_context"]
    chunk_size = config["encoder"]["chunk_size"]
    pre_enc = config["encoder"]["pre_encode_cache_size"]

    batch = 1
    time_steps = chunk_size + pre_enc

    # Cache format is [batch, n_layers, ...] (forward_for_export transposes internally)
    dummy_signal = torch.randn(batch, 128, time_steps)
    dummy_length = torch.tensor([time_steps], dtype=torch.long)
    dummy_cache_channel = torch.zeros(batch, n_layers, left_context, d_model)
    dummy_cache_time = torch.zeros(batch, n_layers, d_model, conv_context)
    dummy_cache_len = torch.zeros(1, dtype=torch.long)
    dummy_spk_targets = torch.ones(batch, time_steps)
    dummy_bg_spk_targets = torch.zeros(batch, time_steps)

    input_names = [
        "processed_signal", "processed_signal_length",
        "cache_last_channel", "cache_last_time", "cache_last_channel_len",
        "spk_targets", "bg_spk_targets",
    ]
    output_names = [
        "encoded", "encoded_len",
        "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next",
    ]
    dynamic_axes = {
        "processed_signal": {2: "time"},
        "spk_targets": {1: "spk_time"},
        "bg_spk_targets": {1: "spk_time"},
        "encoded": {2: "encoded_time"},
    }

    print(f"  Signal shape: [1, 128, {time_steps}], cache: [{batch}, {n_layers}, {left_context}, {d_model}]")
    print(f"  Exporting encoder with {len(input_names)} inputs...")

    torch.onnx.export(
        wrapper,
        (dummy_signal, dummy_length, dummy_cache_channel, dummy_cache_time,
         dummy_cache_len, dummy_spk_targets, dummy_bg_spk_targets),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify spk_targets wasn't constant-folded out of the graph
    print("  Consolidating encoder weights into single .data file...")
    model_proto = onnx.load(output_path)

    input_names_exported = [inp.name for inp in model_proto.graph.input]
    if "spk_targets" not in input_names_exported:
        print("  WARNING: spk_targets was constant-folded out of the ONNX graph!")
        raise RuntimeError("spk_targets constant-folded; falling back to alternative export")

    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    onnx.save(
        model_proto, output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
    )

    # Clean up scattered weight files from the initial export
    output_dir = os.path.dirname(output_path)
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath) and f.startswith(
            ("onnx__MatMul_", "encoder.", "Constant_", "spk_kernels.", "bg_spk_kernels.")
        ):
            if not f.endswith((".onnx", ".onnx.data")):
                os.remove(fpath)

    print(f"  encoder.onnx: {os.path.getsize(output_path) / (1024*1024):.1f}MB"
          f" + {os.path.getsize(data_file) / (1024*1024):.1f}MB data")


def fallback_export_encoder(asr_model, output_path, config):
    """Fallback: remove hooks and replicate kernel injection explicitly in forward().

    More reliable for ONNX tracing because all operations are explicit rather
    than relying on hooks, but requires duplicating the encoder's layer loop.
    """
    d_model = config["encoder"]["d_model"]
    n_layers = config["encoder"]["num_layers"]
    left_context = config["encoder"]["left_context"]
    conv_context = config["encoder"]["conv_context"]
    chunk_size = config["encoder"]["chunk_size"]
    pre_enc = config["encoder"]["pre_encode_cache_size"]
    spk_kernel_layers = config["speaker_kernels"]["spk_kernel_layers"]

    class ExplicitKernelEncoder(nn.Module):
        """Encoder wrapper with explicit kernel injection (no hooks)."""

        def __init__(self, encoder, spk_kernels, bg_spk_kernels, kernel_layers):
            super().__init__()
            self.encoder = encoder
            self.spk_kernels = spk_kernels
            self.bg_spk_kernels = bg_spk_kernels
            self.kernel_layers = [str(x) for x in kernel_layers]

            # Remove all hooks so they don't interfere
            for layer in self.encoder.layers:
                layer._forward_hooks.clear()
                layer._forward_pre_hooks.clear()
            self.encoder._forward_hooks.clear()
            self.encoder._forward_pre_hooks.clear()

        def solve_length_mismatch(self, x, mask, default_value=1.0):
            if mask.shape[1] < x.shape[1]:
                mask = torch.nn.functional.pad(
                    mask, (x.shape[1] - mask.shape[1], 0),
                    mode="constant", value=default_value,
                )
            elif mask.shape[1] > x.shape[1]:
                mask = mask[:, -x.shape[1]:]
            return mask

        def forward(
            self, processed_signal, processed_signal_length,
            cache_last_channel, cache_last_time, cache_last_channel_len,
            spk_targets, bg_spk_targets,
        ):
            enc = self.encoder

            # Transpose caches: [batch, layers, ...] -> [layers, batch, ...]
            cache_last_channel_t = cache_last_channel.transpose(0, 1)
            cache_last_time_t = cache_last_time.transpose(0, 1)

            audio_signal = torch.transpose(processed_signal, 1, 2)
            audio_signal, length = enc.pre_encode(x=audio_signal, lengths=processed_signal_length)
            length = length.to(torch.int64)

            if enc.streaming_cfg.drop_extra_pre_encoded > 0:
                audio_signal = audio_signal[:, enc.streaming_cfg.drop_extra_pre_encoded:, :]
                length = (length - enc.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

            max_audio_length = audio_signal.size(1)
            cache_len = enc.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - enc.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len

            audio_signal, pos_emb = enc.pos_enc(x=audio_signal, cache_len=cache_len)

            pad_mask, att_mask = enc._create_masks(
                att_context_size=enc.att_context_size,
                padding_length=padding_length,
                max_audio_length=max_audio_length,
                offset=offset,
                device=audio_signal.device,
            )
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]

            cache_last_time_next = []
            cache_last_channel_next = []

            for lth, (_, layer) in enumerate(zip(enc.layer_drop_probs, enc.layers)):
                # Inject speaker kernels before the layer (replicating the hook)
                layer_idx_str = str(lth)
                if layer_idx_str in self.kernel_layers:
                    spk_mask = self.solve_length_mismatch(audio_signal, spk_targets, default_value=1.0)
                    x_spk = self.spk_kernels[layer_idx_str](audio_signal * spk_mask.unsqueeze(2))
                    audio_signal = audio_signal + x_spk

                    bg_mask = self.solve_length_mismatch(audio_signal, bg_spk_targets, default_value=0.0)
                    x_bg = self.bg_spk_kernels[layer_idx_str](audio_signal * bg_mask.unsqueeze(2))
                    audio_signal = audio_signal + x_bg

                audio_signal = layer(
                    x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
                    cache_last_channel=cache_last_channel_t[lth],
                    cache_last_time=cache_last_time_t[lth],
                )
                audio_signal, ch_next, t_next = audio_signal
                cache_last_channel_next.append(ch_next)
                cache_last_time_next.append(t_next)

            if enc.out_proj is not None:
                audio_signal = enc.out_proj(audio_signal)

            audio_signal = torch.transpose(audio_signal, 1, 2)
            length = length.to(dtype=torch.int64)

            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)

            rets = enc.streaming_post_process(
                (audio_signal, length, cache_last_channel_next, cache_last_time_next,
                 torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len)),
                keep_all_outputs=False,
            )

            return (rets[0], rets[1],
                    rets[2].transpose(0, 1), rets[3].transpose(0, 1), rets[4])

    wrapper = ExplicitKernelEncoder(
        asr_model.encoder, asr_model.spk_kernels,
        asr_model.bg_spk_kernels, spk_kernel_layers,
    )
    wrapper.eval()

    batch = 1
    time_steps = chunk_size + pre_enc

    dummy_signal = torch.randn(batch, 128, time_steps)
    dummy_length = torch.tensor([time_steps], dtype=torch.long)
    dummy_cache_channel = torch.zeros(batch, n_layers, left_context, d_model)
    dummy_cache_time = torch.zeros(batch, n_layers, d_model, conv_context)
    dummy_cache_len = torch.zeros(1, dtype=torch.long)
    dummy_spk_targets = torch.ones(batch, time_steps)
    dummy_bg_spk_targets = torch.zeros(batch, time_steps)

    input_names = [
        "processed_signal", "processed_signal_length",
        "cache_last_channel", "cache_last_time", "cache_last_channel_len",
        "spk_targets", "bg_spk_targets",
    ]
    output_names = [
        "encoded", "encoded_len",
        "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next",
    ]
    dynamic_axes = {
        "processed_signal": {2: "time"},
        "spk_targets": {1: "spk_time"},
        "bg_spk_targets": {1: "spk_time"},
        "encoded": {2: "encoded_time"},
    }

    print(f"  Exporting encoder (fallback, explicit kernel injection) with {len(input_names)} inputs...")
    torch.onnx.export(
        wrapper,
        (dummy_signal, dummy_length, dummy_cache_channel, dummy_cache_time,
         dummy_cache_len, dummy_spk_targets, dummy_bg_spk_targets),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    model_proto = onnx.load(output_path)
    input_names_exported = [inp.name for inp in model_proto.graph.input]
    if "spk_targets" not in input_names_exported:
        print("  WARNING: spk_targets was still constant-folded out!")
    else:
        print("  spk_targets confirmed as ONNX graph input")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  encoder.onnx: {size_mb:.1f}MB")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Export multitalker Parakeet streaming model to ONNX"
    )
    parser.add_argument(
        "--nemo-path", default=DEFAULT_NEMO_PATH,
        help="Path to .nemo model file",
    )
    parser.add_argument(
        "--no-quantise", action="store_true",
        help="Export fp32 models only, skip quantisation",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory (default: output)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.nemo_path):
        print(f"Model not found: {args.nemo_path}")
        print("Download from: https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.nemo_path}...")
    asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_path, map_location="cpu")
    asr_model.eval()

    config = get_model_config(asr_model)

    config_path = os.path.join(args.output_dir, "multitalker_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Saved config to {config_path}")

    # Extract tokenizer
    print("\nExtracting tokenizer...")
    extract_tokenizer(args.nemo_path, args.output_dir)

    # Export encoder (try hook-based first, fall back to explicit kernel injection)
    encoder_path = os.path.join(args.output_dir, "encoder.onnx")
    print("\nExporting encoder...")
    try:
        custom_export_encoder(asr_model, encoder_path, config)
    except Exception as e:
        print(f"  Primary export failed: {e}")
        traceback.print_exc()
        print("\n  Falling back to explicit kernel injection export...")
        asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_path, map_location="cpu")
        asr_model.eval()
        fallback_export_encoder(asr_model, encoder_path, config)

    # Export decoder + joint
    decoder_path = os.path.join(args.output_dir, "decoder_joint.onnx")
    print("\nExporting decoder_joint...")
    # Reload if hooks were removed during fallback export
    if not hasattr(asr_model.encoder.layers[0], "_forward_pre_hooks") or \
       len(asr_model.encoder.layers[0]._forward_pre_hooks) == 0:
        print("  Reloading model (hooks were removed during encoder export)...")
        asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_path, map_location="cpu")
        asr_model.eval()
    export_decoder_joint(asr_model, decoder_path, config)

    # Print ONNX I/O summary
    print("\nEncoder ONNX inputs/outputs:")
    model_proto = onnx.load(encoder_path)
    for inp in model_proto.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  input  {inp.name}: {dims}")
    for out in model_proto.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  output {out.name}: {dims}")

    print("\nDecoder ONNX inputs/outputs:")
    dec_proto = onnx.load(decoder_path)
    for inp in dec_proto.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  input  {inp.name}: {dims}")
    for out in dec_proto.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  output {out.name}: {dims}")

    # Validate
    print("\nValidating ONNX models...")
    onnx.checker.check_model(encoder_path)
    print("  encoder.onnx: OK")
    onnx.checker.check_model(decoder_path)
    print("  decoder_joint.onnx: OK")

    # Metadata
    dec = asr_model.decoder
    meta_data = {
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

    if args.no_quantise:
        print("\nSkipping quantisation (--no-quantise)")
        add_meta_data(encoder_path, meta_data)
        add_meta_data(decoder_path, meta_data)
    else:
        dynamic_quantise(args.output_dir, meta_data)

    # Summary
    print("\nOutput files:")
    for f in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size > 1024 * 1024:
                print(f"  {f}: {size / (1024*1024):.1f}MB")
            elif size > 1024:
                print(f"  {f}: {size / 1024:.1f}KB")
            else:
                print(f"  {f}: {size}B")


if __name__ == "__main__":
    main()
