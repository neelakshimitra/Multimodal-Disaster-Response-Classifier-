# Multimodal Fusion for Humanitarian Needs Classification

**ST311 Deep Learning Project — Group 8**

Combines xBD post-disaster satellite imagery with CrisisMMD crisis tweets to classify humanitarian needs into four categories (infrastructure, medical, rescue, other), comparing five model architectures and testing whether satellite imagery adds signal beyond text alone.

## Data
- **xBD**: post-disaster satellite imagery (Hurricane Harvey, 2017 Mexico Earthquake)
- **CrisisMMD**: crisis tweets with humanitarian-need labels (Alam et al., 2018)
- Datasets are aligned at the **event level only** (not sample-level) — each tweet is paired with a randomly sampled satellite image from the same disaster event
- After filtering: **998 training, 166 validation, 163 test** tweet-image pairs
- Heavily imbalanced: rescue is 42.5% of training data; medical is only 23 training / 2 test examples

## Method
Five architectures compared, all using ResNet-18 (image) and DistilBERT (text) as encoders:
1. **CNN Baseline** — image only
2. **Text Baseline** — DistilBERT text only
3. **Late Fusion** — independent encoding, concatenation before classification
4. **Early Fusion** — projected embeddings combined via element-wise addition
5. **Attention Fusion** — bidirectional cross-attention between modalities

Trained with weighted cross-entropy (to handle class imbalance), Adam optimizer with differential learning rates (1e-5 pretrained layers, 1e-3 new layers), 10 epochs, evaluated via macro F1.

## Key Results
| Model | Macro F1 |
|---|---|
| **Attention Fusion** | **0.8168** |
| Late Fusion | 0.7684 |
| Early Fusion | 0.7301 |
| Text Baseline | 0.6505 |
| CNN Baseline | 0.2271 |

- Attention Fusion is the best-performing model overall, beating the strongest unimodal baseline (text) by 0.167 macro F1.
- **Missing-modality ablation**: removing text drops macro F1 by 0.62–0.64 across all fusion models — roughly **10x larger** than removing images (0.00–0.06 drop). Text is the dominant signal; satellite imagery contributes only a modest, architecture-dependent supplement (largest under attention fusion).
- This points to **weak event-level alignment** (rather than lack of informative imagery) as the key bottleneck limiting the image branch's contribution — a direction for future work using coordinate/timestamp-level alignment.

## Tools
Python, PyTorch, Torchvision, Hugging Face Transformers (DistilBERT), scikit-learn, Google Colab (T4 GPU).

## Paper
Full write-up with methodology, related work, and appendix figures included in the repo.
