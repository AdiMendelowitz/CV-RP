# Architecture Benchmark — Linear Probe on CIFAR-10

**Protocol:** Frozen pretrained backbone (ImageNet weights via timm) + logistic regression (C=0.316, L2-normalised features). Input resized to 224×224. Inference time: median over 200 runs, batch=1, CPU.

| Model | Top-1 Acc (%) | Params (M) | Inference (ms/img) |
|-------|:-------------:|:----------:|:------------------:|
| ResNet-18 | 83.83 | 11.2 | 26.8 |
| ViT-Tiny | 80.72 | 5.5 | 26.9 |
| EfficientNet-B0 | 90.06 | 4.0 | 29.2 |
| ConvNeXt-Tiny | 95.08 | 27.8 | 90.6 |

## Notes

- Accuracy reflects **representation quality** of the pretrained backbone, not fine-tuned performance on CIFAR-10.
- ViT-Tiny is expected to underperform CNNs here: attention lacks the locality inductive bias that benefits small datasets.
- Inference time covers the backbone forward pass only (no logistic regression head).
