# MAEST Discogs 10s Model Card

## Model

- Name: `discogs-maest-10s-pw-129e`
- Provider: Music Technology Group, Universitat Pompeu Fabra
- Source: <https://huggingface.co/mtg-upf/discogs-maest-10s-pw-129e>
- Architecture: Music Audio Efficient Spectrogram Transformer (MAEST)
- Parameters: 85,994,128
- Runtime: Core ML ML Program, fp16
- Analyzer version: 1
- Released model version: 1.0.0

## Input and output

- 16 kHz mono audio
- 10 second window
- 96 mel bins
- 626 frames
- Input tensor: `mel`, float32, shape `[1, 626, 96]`
- Output tensor: `logits`, 400 Discogs style scores

Lyrid aggregates three windows at 20%, 50% and 80% of a track, applies
sigmoid to the logits, then keeps style and atmosphere labels at or above the
configured confidence threshold.

## Conversion

The published artifact is converted from the MTG PyTorch checkpoint with:

- PyTorch 2.2.2 and Transformers 4.46.3
- Core ML Tools 9.0
- TorchScript tracing with fixed input shape `[1, 626, 96]`
- Core ML fp16 compute precision and macOS 14 deployment target

The original feature extractor's mel filterbank is exported as a float32
matrix for Lyrid's Accelerate implementation. The model directory itself does
not contain the original PyTorch weights.

## License

The pretrained weights and converted artifacts are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Commercial use requires a proprietary license from MTG / UPF. The model is
not included in the Lyrid application bundle.

## Attribution

Music Technology Group, Universitat Pompeu Fabra; Pablo Alonso-Jimenez,
Xavier Serra, Dmitry Bogdanov.

Paper: <https://arxiv.org/abs/2309.16418>

## Limitations

- Labels are Discogs style categories, not a validated mood taxonomy.
- Heavy re-encoding may change the exact audio fingerprint and require
  re-analysis.
- The model does not reliably detect speech-only or non-music audio.
- This model card is not a legal opinion.

