#!/usr/bin/env bash
# Check that the burn backend gives the same output as ONNX Runtime for every model present in
# the repo root. Run it after upgrading burn or regenerating src/burn_backend/generated.
#
# Usage: scripts/check_burn_parity.sh [audio.wav]     (default: 6_speakers.wav)
#
# Each model runs on ONNX Runtime (CPU), burn CPU and burn GPU (Metal on macOS, wgpu elsewhere)
# through examples/burn.rs; the transcripts or speaker segments must be identical. Models whose
# files are missing are skipped. Exits non-zero if any output differs or a run fails.

set -u
cd "$(dirname "$0")/.."

audio="${1:-6_speakers.wav}"
gpu_feature=$([ "$(uname)" = "Darwin" ] && echo metal || echo wgpu)
diar=nemotron3_diar_v3.onnx

echo "building examples/burn (ort + $gpu_feature)..."
cargo build --release --quiet --example burn --features "$gpu_feature,multitalker,cohere" || exit 1
bin=target/release/examples/burn
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

failed=0
# name | model kind | model path | extra argument | files that must exist
check() {
    local name=$1 kind=$2 path=$3 extra=$4
    shift 4
    for f in "$@"; do
        if [ ! -e "$f" ]; then
            printf "%-16s skipped (no %s)\n" "$name" "$f"
            return
        fi
    done
    local line="" device
    for device in ort cpu gpu; do
        # shellcheck disable=SC2086
        if ! "$bin" "$kind" "$audio" "$path" "$device" $extra > "$out/$name-$device.txt" 2>&1; then
            printf "%-16s FAILED on %s: %s\n" "$name" "$device" "$(tail -1 "$out/$name-$device.txt")"
            failed=1
            return
        fi
    done
    for device in cpu gpu; do
        if diff <(grep '^\[' "$out/$name-ort.txt") <(grep '^\[' "$out/$name-$device.txt") > /dev/null; then
            line="$line $device=same"
        else
            line="$line $device=DIFFERENT"
            failed=1
        fi
    done
    printf "%-16s%s\n" "$name" "$line"
}

check tdt            tdt         tdt            ""          tdt/encoder-model.onnx
check parakeet-ultra tdt         parakeet-ultra ""          parakeet-ultra/encoder-model.onnx
check ctc            ctc         ctc            ""          ctc/model.onnx ctc/model.onnx_data
check unified        unified     unified        ""          unified/encoder.onnx unified/encoder.onnx.data
check eou            eou         eou            ""          eou/encoder.onnx
check nemotron       nemotron    nemotron       ""          nemotron/encoder.onnx
check nemotron-multi nemotron    nemotron_multi ""          nemotron_multi/encoder.onnx
check multitalker    multitalker multitalker    "$diar"     multitalker/encoder.onnx "$diar"
check diar-offline   diar        "$diar"        offline     "$diar"
check diar-low       diar        "$diar"        low         "$diar"
check cohere         cohere      cohere         ""          cohere/encoder_model.onnx cohere/decoder_model_merged.onnx

exit $failed
