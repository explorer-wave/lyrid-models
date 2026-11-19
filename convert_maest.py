#!/usr/bin/env python3
"""Convert the MTG MAEST checkpoint into Lyrid's Core ML package.

The Lyrid runtime expects a fixed normalized mel input and a 400-class logits
output. The source Hugging Face model already consumes mel features through
its AST encoder, so the conversion only needs a small wrapper plus a scripted
trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForAudioClassification
from transformers.audio_utils import mel_filter_bank


DEFAULT_MODEL_ID = "mtg-upf/discogs-maest-10s-pw-129e"
DEFAULT_VERSION = "1.0.0"
EXPECTED_LABELS_SHA256 = (
    "bdfe4e41df1987ba2387750207b77df089c90c37d8064e2d6b7dd93e30800e72"
)
EXPECTED_FILTERBANK_SHA256 = (
    "d046cc5a5fea8ec56892f73c5d7c7c78c48ca410a8ea54c29e37da5658451096"
)
MODEL_DIRECTORY_NAME = "maest-10s-fp16.mlmodelc"
PACKAGE_NAME = "maest-10s-fp16.mlpackage"
LABELS_FILE_NAME = "labels.json"
FILTERBANK_FILE_NAME = "maest_mel_filters.bin"


class MAESTLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        output = self.model(mel, return_dict=False)
        return output[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAEST to a Lyrid Core ML model package."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID or local model directory "
        f"(default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Model version written into the manifest "
        f"(default: {DEFAULT_VERSION}).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output model directory. It must not already exist.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def normalized_labels(model: torch.nn.Module) -> list[str]:
    id2label = model.config.id2label
    labels = [id2label[index] for index in sorted(id2label, key=int)]
    if len(labels) != 400:
        raise SystemExit(f"expected 400 labels, got {len(labels)}")
    if len(set(labels)) != len(labels):
        raise SystemExit("model label map contains duplicates")
    return labels


def write_filterbank(path: Path) -> None:
    filters = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=96,
        min_frequency=0.0,
        max_frequency=8_000.0,
        sampling_rate=16_000,
        norm="slaney",
        mel_scale="slaney",
    )
    if filters.shape != (257, 96):
        raise SystemExit(f"unexpected filterbank shape: {filters.shape}")
    path.write_bytes(np.asarray(filters, dtype="<f4").tobytes(order="C"))


def convert(
    model_id: str,
    version: str,
    output: Path,
) -> None:
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    output.mkdir(parents=True)
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    labels = normalized_labels(model)
    labels_path = output / LABELS_FILE_NAME
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if sha256(labels_path) != EXPECTED_LABELS_SHA256:
        raise SystemExit(
            "generated labels.json does not match the reference label map"
        )

    filterbank_path = output / FILTERBANK_FILE_NAME
    write_filterbank(filterbank_path)
    if sha256(filterbank_path) != EXPECTED_FILTERBANK_SHA256:
        raise SystemExit(
            "generated filterbank does not match the MAEST reference values"
        )

    wrapper = MAESTLogits(model).eval()
    example = torch.randn(1, 626, 96, dtype=torch.float32)
    with torch.no_grad():
        expected = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False)
        actual = traced(example)
    max_difference = float((expected - actual).abs().max())
    if not np.isfinite(max_difference) or max_difference > 1e-4:
        raise SystemExit(
            "TorchScript trace changed model output "
            f"(max difference {max_difference:.6g})"
        )

    with tempfile.TemporaryDirectory(prefix="maest-coreml-") as temporary:
        package_path = Path(temporary) / PACKAGE_NAME
        coreml_model = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="mel",
                    shape=(1, 626, 96),
                    dtype=np.float32,
                )
            ],
            outputs=[ct.TensorType(name="logits")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS17,
        )
        coreml_model.short_description = (
            "MAEST Discogs 10s style classifier for Lyrid Player"
        )
        coreml_model.author = (
            "Music Technology Group, Universitat Pompeu Fabra; "
            "converted for Lyrid"
        )
        coreml_model.license = "CC BY-NC-SA 4.0"
        coreml_model.user_defined_metadata.update(
            {
                "lyrid.modelKey": "discogs-maest-10s-pw-129e",
                "lyrid.analyzerVersion": "1",
                "lyrid.sourceURL": (
                    "https://huggingface.co/"
                    "mtg-upf/discogs-maest-10s-pw-129e"
                ),
                "lyrid.conversion": (
                    "PyTorch 2.2.2 / Transformers 4.46.3 / "
                    "Core ML Tools 9.0"
                ),
            }
        )
        coreml_model.save(package_path)
        subprocess.run(
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                str(package_path),
                str(output),
            ],
            check=True,
        )

    model_directory = output / MODEL_DIRECTORY_NAME
    if not model_directory.is_dir():
        raise SystemExit(f"compiler did not produce {model_directory}")

    coreml_files: dict[str, dict[str, object]] = {}
    for path in sorted(model_directory.rglob("*")):
        if path.is_file():
            relative = path.relative_to(output).as_posix()
            coreml_files[relative] = file_record(path)
    coreml_files[LABELS_FILE_NAME] = file_record(labels_path)
    coreml_files[FILTERBANK_FILE_NAME] = file_record(filterbank_path)

    manifest = {
        "model": "discogs-maest-10s-pw-129e",
        "version": version,
        "format": "coreml-fp16",
        "source": (
            "https://huggingface.co/"
            "mtg-upf/discogs-maest-10s-pw-129e"
        ),
        "license": "CC BY-NC-SA 4.0",
        "licenseURL": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "commercialUse": (
            "A proprietary license is required from MTG / UPF."
        ),
        "attribution": (
            "Music Technology Group, Universitat Pompeu Fabra; "
            "Pablo Alonso-Jimenez, Xavier Serra, Dmitry Bogdanov"
        ),
        "provider": "MAEST",
        "displayName": "MAEST · Discogs 10s",
        "minimumAppVersion": "1.0.0",
        "minimumMacOSVersion": "14.0",
        "analyzerVersion": 1,
        "input": {
            "sampleRate": 16000,
            "channels": 1,
            "windowSeconds": 10,
            "melBins": 96,
            "maxFrames": 626,
        },
        "output": {
            "classes": 400,
            "scores": "sigmoid probabilities",
        },
        "coreML": {
            "format": "coreml-fp16",
            "directory": MODEL_DIRECTORY_NAME,
            "labels": LABELS_FILE_NAME,
            "filterbank": FILTERBANK_FILE_NAME,
            "files": coreml_files,
        },
    }
    manifest_path = output / "model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Model directory: {output}")
    print(f"Manifest:        {manifest_path}")
    print(f"Labels SHA-256:  {sha256(labels_path)}")
    print(f"Filterbank:      {sha256(filterbank_path)}")
    print(f"Trace max diff:  {max_difference:.6g}")


def main() -> None:
    args = parse_args()
    convert(args.model_id, args.version, args.output)


if __name__ == "__main__":
    main()
