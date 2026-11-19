MAEST Discogs 10s Core ML model for Lyrid Player.

- Model: `discogs-maest-10s-pw-129e`
- Runtime: Core ML ML Program, fp16
- Input: `mel`, float32, `[1, 626, 96]` (16 kHz mono, 10 seconds)
- Output: `logits`, 400 Discogs style scores
- Analyzer version: 1
- License: CC BY-NC-SA 4.0

Assets:

- `manifest.json`
- `maest-10s-fp16-1.0.0.zip`

The ZIP contains `maest-10s-fp16.mlmodelc`, `labels.json` and
`maest_mel_filters.bin`. The manifest records the archive and per-file
SHA-256 hashes.

Attribution: Music Technology Group, Universitat Pompeu Fabra; Pablo
Alonso-Jimenez, Xavier Serra, Dmitry Bogdanov. Commercial use requires a
proprietary license from MTG / UPF.
