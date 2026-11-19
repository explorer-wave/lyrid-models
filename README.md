# Lyrid model releases

This repository contains the maintained conversion pipeline, packaging tools,
model cards and release metadata for the optional automatic-tagging models
used by [Lyrid Player](https://github.com/explorer-wave/musicbee-mac).

Model weights are not committed to Git. Each GitHub Release publishes:

- `manifest.json`: the schema-v1 model manifest consumed by Lyrid.
- `maest-10s-fp16-<version>.zip`: the compiled Core ML model, labels and mel
  filterbank.

Git tags and manifest versions must match exactly. Lyrid only accepts stable
semantic versions and requires the `manifest.json` asset on every release.

## Current model

- Source: [`mtg-upf/discogs-maest-10s-pw-129e`](https://huggingface.co/mtg-upf/discogs-maest-10s-pw-129e)
- Architecture: Music Audio Efficient Spectrogram Transformer (MAEST)
- Input: 16 kHz mono, 10 seconds, 96 mel bins, 626 frames
- Output: 400 Discogs style logits
- Runtime format: Core ML ML Program, fp16

The first published version is `1.0.0`; its checked-in release manifest is
[`manifest.json`](manifest.json).

## Conversion

The conversion script requires macOS, Xcode command-line tools, Python 3.12
and the pinned packages in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python convert_maest.py \
  --version 1.0.0 \
  --output build/1.0.0
```

The script:

1. Loads the MTG checkpoint and its label map.
2. Wraps the PyTorch classifier so its input is the normalized mel tensor
   `[1, 626, 96]` expected by Lyrid.
3. Traces the model and converts it to a Core ML ML Program in fp16.
4. Compiles the `.mlpackage` with `xcrun coremlcompiler`.
5. Generates `labels.json`, `maest_mel_filters.bin` and the source
   `model-manifest.json` used by the release packager.

The generated filterbank is checked against the MAEST reference values. The
script does not upload anything and does not publish a release.

## Packaging a release

```bash
bash package-model-release.sh \
  --model-dir build/1.0.0 \
  --version 1.0.0 \
  --output dist/1.0.0
```

The packager verifies every declared file size and SHA-256, rejects symlinks,
cache files and unsafe archive paths, creates the ZIP, and writes the release
manifest next to it. Upload both files to a GitHub Release whose tag is
`1.0.0`.

## Consuming a release

Lyrid reads the public Release list and downloads `manifest.json` from the
matching release. The app then streams the ZIP, verifies its size and SHA-256,
checks every extracted file, runs a Core ML smoke test and installs the model
atomically.

## License and attribution

The pretrained weights are distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Commercial use requires a separate proprietary license from MTG / UPF.

The model is a conversion of the MTG-published PyTorch weights to Core ML.
Attribution: Music Technology Group, Universitat Pompeu Fabra; Pablo
Alonso-Jimenez, Xavier Serra, Dmitry Bogdanov.

See [`LICENSE`](LICENSE) and [`MODEL_CARD.md`](MODEL_CARD.md). This repository
does not grant rights beyond the upstream model license.
