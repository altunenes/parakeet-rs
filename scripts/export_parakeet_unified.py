#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Any, cast

import onnx
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

import nemo.collections.asr as nemo_asr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "nemo" / "parakeet-unified-en-0.6b.nemo"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "nemo" / "onnx_export"
DEFAULT_WORK_DIR = REPO_ROOT / "nemo" / ".export_work_unified"

UNSUPPORTED_ENCODER_KEYS = {
    "att_chunk_context_size",
    "att_chunk_use_dynamic_chunking",
    "att_mask_style",
    "att_zero_rc_weight",
    "conv_context_style",
}

REQUIRED_ENCODER_INPUTS = {"audio_signal", "length"}
REQUIRED_ENCODER_OUTPUTS = {"outputs", "encoded_lengths"}
REQUIRED_DECODER_INPUTS = {
    "encoder_outputs",
    "targets",
    "target_length",
    "input_states_1",
    "input_states_2",
}
REQUIRED_DECODER_OUTPUTS = {"outputs", "output_states_1", "output_states_2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NVIDIA Parakeet Unified RNNT ASR to ONNX"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the unified .nemo model",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for ONNX outputs",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="Temporary work directory for raw export artifacts",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Also generate dynamically quantized int8 encoder/decoder artifacts",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary work directory after export for debugging",
    )
    return parser.parse_args()


def configure_nemo_logging() -> None:
    logging.getLogger("nemo_logging").setLevel(logging.ERROR)
    try:
        from nemo.core.classes.common import typecheck

        typecheck.set_typecheck_enabled(False)
    except ImportError:
        pass


def patch_torch_onnx_export() -> None:
    pytorch_version = tuple(
        int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
    )
    patch_marker = "_legacy_onnx_patched"
    print(f"PyTorch version: {torch.__version__}")

    if pytorch_version < (2, 9) or getattr(torch.onnx.export, patch_marker, False):
        return

    print("Patching torch.onnx.export for PyTorch 2.9+ (dynamo=False)")
    original_export = torch.onnx.export

    @functools.wraps(original_export)
    def patched_export(*pargs, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original_export(*pargs, **kwargs)

    setattr(patched_export, patch_marker, True)
    torch.onnx.export = patched_export


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_archive_member(archive: tarfile.TarFile, suffix: str) -> str:
    for name in archive.getnames():
        normalized = name.lstrip("./")
        if normalized.endswith(suffix):
            return name
    raise FileNotFoundError(f"Could not find archive member ending with {suffix!r}")


def extract_member(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise RuntimeError(f"Failed to extract {member_name}")

    ensure_parent_dir(destination)
    with destination.open("wb") as out:
        shutil.copyfileobj(extracted, out)


def prepare_override_config(model_path: Path, work_dir: Path) -> Path:
    print("Preparing unified encoder override config...")
    with tarfile.open(model_path, "r:*") as archive:
        config_member = find_archive_member(archive, "model_config.yaml")
        extracted = archive.extractfile(config_member)
        if extracted is None:
            raise RuntimeError("Failed to read model_config.yaml from .nemo archive")
        config = yaml.safe_load(extracted)

    encoder_cfg = config["encoder"]
    for key in UNSUPPORTED_ENCODER_KEYS:
        encoder_cfg.pop(key, None)

    encoder_cfg["att_context_style"] = "regular"
    encoder_cfg["att_context_size"] = [-1, -1]

    override_path = work_dir / "override_model_config.yaml"
    ensure_parent_dir(override_path)
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return override_path


def extract_tokenizer_artifacts(model_path: Path, output_dir: Path) -> None:
    print("Extracting tokenizer artifacts...")
    with tarfile.open(model_path, "r:*") as archive:
        member_map = {
            "tokenizer.model": "tokenizer.model",
            "vocab.txt": "vocab.txt",
        }

        for archive_suffix, output_name in member_map.items():
            try:
                member_name = find_archive_member(archive, archive_suffix)
            except FileNotFoundError:
                continue
            extract_member(archive, member_name, output_dir / output_name)
            print(f"  Extracted {output_name}")


def find_single(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one match for {pattern!r}, found {len(matches)}: {matches}"
        )
    return matches[0]


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def save_single_file_onnx(src: Path, dest: Path) -> None:
    ensure_parent_dir(dest)
    model = onnx.load_model(str(src), load_external_data=True)
    remove_if_exists(dest)
    onnx.save_model(model, str(dest), save_as_external_data=False)


def save_external_data_onnx(src: Path, dest: Path, data_filename: str) -> None:
    ensure_parent_dir(dest)
    model = onnx.load_model(str(src), load_external_data=True)
    data_path = dest.with_name(data_filename)
    remove_if_exists(dest)
    remove_if_exists(data_path)
    onnx.save_model(
        model,
        str(dest),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=0,
    )


def quantize_to_tmp(src: Path, tmp_dest: Path, weight_type: QuantType) -> Path:
    tmp_dest.parent.mkdir(parents=True, exist_ok=True)
    for existing in tmp_dest.parent.glob(f"{tmp_dest.name}*"):
        existing.unlink()

    quantize_dynamic(
        model_input=str(src),
        model_output=str(tmp_dest),
        weight_type=weight_type,
    )
    return tmp_dest


def validate_graph_io(model_path: Path, required_inputs: set[str], required_outputs: set[str]) -> None:
    model = onnx.load_model(str(model_path), load_external_data=False)
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}

    missing_inputs = sorted(required_inputs - input_names)
    missing_outputs = sorted(required_outputs - output_names)

    if missing_inputs or missing_outputs:
        raise RuntimeError(
            f"Validation failed for {model_path.name}: missing inputs {missing_inputs}, "
            f"missing outputs {missing_outputs}"
        )


def validate_artifacts(output_dir: Path, quantized: bool) -> None:
    print("Validating exported artifacts...")

    encoder_path = output_dir / "encoder.onnx"
    encoder_data_path = output_dir / "encoder.onnx.data"
    decoder_path = output_dir / "decoder_joint.onnx"

    required_paths = [encoder_path, encoder_data_path, decoder_path, output_dir / "tokenizer.model"]
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"Missing required export artifact: {path}")

    validate_graph_io(encoder_path, REQUIRED_ENCODER_INPUTS, REQUIRED_ENCODER_OUTPUTS)
    validate_graph_io(decoder_path, REQUIRED_DECODER_INPUTS, REQUIRED_DECODER_OUTPUTS)

    if quantized:
        quantized_encoder = output_dir / "encoder.int8.onnx"
        quantized_encoder_data = output_dir / "encoder.int8.onnx.data"
        quantized_decoder = output_dir / "decoder_joint.int8.onnx"
        for path in [quantized_encoder, quantized_encoder_data, quantized_decoder]:
            if not path.exists():
                raise RuntimeError(f"Missing quantized export artifact: {path}")

        validate_graph_io(quantized_encoder, REQUIRED_ENCODER_INPUTS, REQUIRED_ENCODER_OUTPUTS)
        validate_graph_io(quantized_decoder, REQUIRED_DECODER_INPUTS, REQUIRED_DECODER_OUTPUTS)


def format_size(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}T"


def print_artifact_summary(output_dir: Path, quantized: bool) -> None:
    print("Export complete:")
    artifact_names = [
        "encoder.onnx",
        "encoder.onnx.data",
        "decoder_joint.onnx",
        "tokenizer.model",
        "vocab.txt",
    ]
    if quantized:
        artifact_names.extend(
            [
                "encoder.int8.onnx",
                "encoder.int8.onnx.data",
                "decoder_joint.int8.onnx",
            ]
        )

    for artifact_name in artifact_names:
        path = output_dir / artifact_name
        if path.exists():
            print(f"  {artifact_name:<24} {format_size(path.stat().st_size)}")


def export_model(model_path: Path, output_dir: Path, work_dir: Path, quantize_int8: bool) -> None:
    raw_export_dir = work_dir / "raw_export"
    quantized_dir = work_dir / "quantized"

    ensure_clean_dir(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    override_config = prepare_override_config(model_path, work_dir)

    print(f"Loading model from: {model_path}")
    model = cast(
        Any,
        nemo_asr.models.ASRModel.restore_from(
        restore_path=str(model_path),
        override_config_path=str(override_config),
        map_location=torch.device("cpu"),
        ),
    )
    model.eval()
    model.freeze()
    model = model.to("cpu")

    print(f"  Model class : {type(model).__name__}")
    print(f"  Encoder type: {type(model.encoder).__name__}")

    print(f"Exporting raw ONNX graphs to: {raw_export_dir}")
    raw_export_dir.mkdir(parents=True, exist_ok=True)
    model.export(str(raw_export_dir / "model.onnx"))

    raw_encoder = find_single(raw_export_dir, "encoder-*.onnx")
    raw_decoder_joint = find_single(raw_export_dir, "decoder_joint-*.onnx")

    print("Normalizing ONNX artifact filenames...")
    save_external_data_onnx(raw_encoder, output_dir / "encoder.onnx", "encoder.onnx.data")
    save_single_file_onnx(raw_decoder_joint, output_dir / "decoder_joint.onnx")

    if quantize_int8:
        print("Quantizing exported models to int8...")
        quantized_encoder = quantize_to_tmp(
            raw_encoder,
            quantized_dir / "encoder.int8.tmp.onnx",
            QuantType.QUInt8,
        )
        quantized_decoder_joint = quantize_to_tmp(
            raw_decoder_joint,
            quantized_dir / "decoder_joint.int8.tmp.onnx",
            QuantType.QInt8,
        )
        save_external_data_onnx(
            quantized_encoder,
            output_dir / "encoder.int8.onnx",
            "encoder.int8.onnx.data",
        )
        save_single_file_onnx(
            quantized_decoder_joint,
            output_dir / "decoder_joint.int8.onnx",
        )

    extract_tokenizer_artifacts(model_path, output_dir)
    validate_artifacts(output_dir, quantize_int8)
    print_artifact_summary(output_dir, quantize_int8)


def main() -> None:
    args = parse_args()
    model_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    configure_nemo_logging()
    patch_torch_onnx_export()

    try:
        export_model(model_path, output_dir, work_dir, args.quantize_int8)
    finally:
        if not args.keep_work_dir and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
