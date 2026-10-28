# Evaluation summary

## Detection quality

| detector | images | instances | tp | fp | fn | precision | recall | f1 | map50 | map50_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 125 | 346 | 10 | 161 | 336 | 0.058 | 0.029 | 0.039 | - | - |
| yolov8n_best | 125 | 346 | - | - | - | 0.741 | 0.628 | - | 0.690 | 0.367 |

The classical baseline emits no confidence score, so it has no PR curve:
mAP is reported as `-` for it by design, not because it failed.
