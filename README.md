# Brugada Syndrome Detection via Graph Neural Networks and Deep Learning

> **IDSC 2026 Submission | Team Whatever | Dataset: Brugada-HUCA (PhysioNet)**

This repository contains the full experimental pipeline for the [IDSC 2026](https://idsc2026.github.io) submission: a comparative study of Histogram Gradient Boosting, ResNet, and Spatial Graph Neural Network architectures for automated binary classification of Brugada syndrome from 12-lead ECG signals using the Brugada-HUCA dataset from [PhysioNet](https://physionet.org/content/brugada-huca/1.0.0).

The codebase is structured to support reproducible hyperparameter search, model comparison across multiple random seeds, and interpretability analysis via lead importance scores and temporal Grad-CAM visualizations.

## Results

The following table reports mean and standard deviation across 5 independent seeds (42 through 46) on the held-out test set. All models use their best hyperparameter configurations from random search. Results are sorted by F2 score, which is the primary evaluation metric.

F2 score weights recall twice as heavily as precision, reflecting the clinical cost asymmetry in Brugada syndrome screening: a missed positive case (false negative) carries a risk of sudden cardiac death, while a false positive results in an unnecessary but non-harmful cardiology referral. Accuracy is included for completeness but is not a meaningful metric given the class imbalance. It is also worth noting that a trivial classifier predicting all negatives would achieve approximately 79% accuracy.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | F2 |
|---|---|---|---|---|---|---|
| ResNet | **0.8255 ± 0.0374** | **0.5853 ± 0.0841** | 0.7833 ± 0.1000 | **0.6630 ± 0.0575** | **0.8713 ± 0.0439** | **0.7282 ± 0.0721** |
| Spatial GNN | 0.7455 ± 0.0905 | 0.4791 ± 0.0958 | **0.8500 ± 0.1616** | 0.5996 ± 0.0866 | 0.8465 ± 0.0728 | 0.7212 ± 0.1039 |
| HGB | 0.7855 ± 0.0506 | 0.5234 ± 0.1104 | 0.6333 ± 0.0408 | 0.5686 ± 0.0724 | 0.7872 ± 0.0340 | 0.6044 ± 0.0496 |

Both deep learning architectures clearly outperform the traditional ML (HGB) baseline. The 1D ResNet achieves the highest overall F2 (0.7282) and AUC (0.8713), demonstrating highly stable and effective temporal feature extraction across the 12 leads.

However, the Spatial GNN achieves the highest absolute Recall (0.8500 vs 0.7833 for ResNet), trading off some precision to cast a wider net. In a screening context where missing a positive case is the most critical failure mode, the Spatial GNN's high sensitivity makes it a highly competitive and clinically relevant alternative.

## Dataset

The project uses the Brugada-HUCA dataset (PhysioNet), which contains 12-lead ECG recordings from 363 patients: 76 diagnosed with Brugada syndrome and 287 normal controls. Signals are sampled at 100 Hz over 12-second, yielding 1200 samples per lead per patient.

The dataset must be downloaded from PhysioNet and placed at the path specified in `configs/base.yml` under `data.path`. The expected directory structure is:

```
<data_path>/
    metadata.csv
    files/
        <patient_id>/
            <patient_id>.hea
            <patient_id>.dat
```

### Data Augmentation

To improve model robustness and simulate real-world signal variations, a data augmentation pipeline is integrated and can be configured in the YAML files under `data.augmentation`.

When `enabled: true`, the following perturbations are applied during training:
*   `amplitude_scale` (e.g., `0.1`): Randomly scales the voltage amplitude of the ECG signal to simulate variations in electrode impedance and heart electrical vector magnitude.
*   `time_shift` (e.g., `0.1`): Randomly shifts the signal along the temporal axis (in seconds) to prevent the model from overfitting to specific alignment points in the 12-second window.
*   `noise_std` (e.g., `0.01`): Injects Gaussian noise to simulate baseline wander, muscle artifacts, and powerline interference.

## Project Structure
```text
.
├── configs/                        # YAML configuration files
│   ├── base.yml                    # Base config inherited by all models
│   ├── best/                       # Best configs saved after hyperparameter search
│   └── *.yml                       # Architecture-specific overrides
├── experiments/                    # Output artifacts, evaluation CSVs, and XAI plots
│   └── physionet.org/files/...     # Default expected download path for the dataset
├── src/
│   ├── data_loader.py              # Dataset class, graph construction, dataloaders
│   ├── interpretability.py         # Lead importance, Grad-CAM, and visualization utilities
│   ├── preprocessing.py            # Signal filtering, normalization, adjacency construction
│   ├── models/                     # Model architecture definitions (ResNet, SpatialGNN, HGB)
│   └── scripts/                    # Automation scripts (hyperparam search, model comparison)
├── main.py                         # Training and evaluation entry point
├── notebook.ipynb                  # Interpretability analysis and visualization
├── pyproject.toml / uv.lock        # Modern Python dependency management
└── requirements.txt                # Fallback dependency list
```

## Installation
The project uses `uv` for lightning-fast dependency management. To install:
```bash
uv sync
```
Alternatively with pip:
```bash
pip install -r requirements.txt
```

Python 3.10 or later is required. The `.python-version` file specifies the exact version used during development.

## Configuration System
All experiments are controlled through YAML configuration files. Model-specific configs (including those in the `configs/best/` directory) inherit from `configs/base.yml` using the `_base_` key, which triggers deep merging at load time — only fields explicitly set in the child config override the base.

The key fields in `configs/base.yml`:
- `data.path`: path to the PhysioNet dataset directory (default: `experiments/physionet.org/files/brugada-huca/1.0.0`)
- `data.augmentation`: block to configure physiological signal perturbations (`amplitude_scale`, `time_shift`, `noise_std`)
- `data.correlation_threshold`: edge pruning threshold for the hybrid adjacency matrix (spatial GNN only)
- `data.anatomic_weight`: weight given to the fixed anatomical adjacency versus the data-driven correlation adjacency (spatial GNN only)
- `training.loss_function`: either `focal` or `bce`; focal loss is used with `alpha=0.79` derived from the class distribution (287/363)
- `tasks`: each task has an `enabled` flag and a `weight` controlling its contribution to the multi-task loss
- `evaluation.primary_metric`: the metric used for early stopping and checkpoint saving; set to `f2` throughout


## Models

### Histogram Gradient Boosting (HGB)
The HGB baseline uses scikit-learn's `HistGradientBoostingClassifier` wrapped in `MultiOutputClassifier` for multi-task support. Because it is a traditional ML model, each output is an independent classifier with no shared representation. Feature extraction is performed by `extract_features()` in `hgb_baseline.py`, which computes handcrafted ECG morphology features per lead: signal statistics (mean, std, max, min), ST-segment elevation, J-point amplitude, ST slope, R-prime wave amplitude, QRS duration proxy, and T-wave polarity.

### 1D ResNet
The ResNet processes all 12 leads simultaneously as a multi-channel 1D signal. The architecture is a stack of `ResNetBlock` modules, each containing two Conv1d layers with BatchNorm and SiLU activation and a skip connection. The first block accepts 12 input channels (one per lead). Subsequent blocks optionally downsample with stride 2, controlled by the `resnet_channels` hyperparameter list. Global average pooling collapses the temporal dimension before the task-specific linear heads.

### Spatial GNN
The SpatialGNN uses a per-lead ResNet encoder to extract temporal embeddings, then applies graph convolution over a lead-level graph where nodes are leads and edges represent inter-lead spatial relationships.

#### Graph Construction

The lead graph is constructed per patient as a weighted combination of two adjacency matrices: a fixed anatomical adjacency and a data-driven correlation adjacency. This hybrid formulation incorporates domain knowledge about electrode placement alongside learned signal correlations to produce clinically grounded graph structures.

The anatomical adjacency encodes the known physical and electrical relationships between the 12 standard leads. The lead ordering used throughout is I, II, III, aVR, aVL, aVF (indices 0-5, limb leads) and V1, V2, V3, V4, V5, V6 (indices 6-11, precordial leads). Connections are defined by two clinical principles.

For the limb leads, edges follow Einthoven's triangle (Goldberger et al., 2013, p. 19), which describes the three standard bipolar leads (I, II, III) as forming a triangle of potential differences around the heart, with aVR, aVL, and aVF derived as augmented unipolar leads from the same electrode set. The specific connections implemented are I-II, I-aVL, II-III, II-aVF, III-aVF, aVR-aVL, aVR-aVF, and aVL-aVF.

For the precordial leads, edges connect sequentially adjacent electrodes (V1-V2, V2-V3, V3-V4, V4-V5, V5-V6), reflecting their physical placement across the chest wall. Goldberger et al. (2013, p. 22) describes V1 through V6 as electrodes placed at six designated positions progressing across the chest wall, with each electrode recording the electrical field at its immediate anatomical location — justifying the sequential edge structure.

The correlation adjacency is computed per patient from the Pearson correlation matrix of the 12 lead signals, capturing dynamic inter-lead relationships that vary with individual cardiac anatomy and pathology. These two matrices are combined as:

```python
hybrid_adj = anatomic_weight * anatomical_adj + (1 - anatomic_weight) * correlation_adj
```

This construction is particularly motivated by the localized nature of Brugada syndrome. The defining ECG pattern — coved-type ST elevation with negative T-wave — is almost exclusively expressed in the right precordial leads V1 and V2. By anchoring the graph structure to known anatomical connectivity, the GNN is guided to attend to the clinically relevant neighbourhood around V1 and V2 rather than discovering it purely from data on a dataset of only 363 patients.

Note that limb leads (indices 0-5) and precordial leads (indices 6-11) have no direct edges in the anatomical adjacency matrix. Cross-group connections can only arise through the correlation component when two leads from different groups exhibit strong Pearson correlation above the threshold. This design choice reflects the anatomical reality that limb leads and precordial leads record fundamentally different projections of the cardiac electrical vector.

#### GNN Architecture

The GNN supports GCN, GAT, and GIN layer types. GCN and GAT use `add_self_loops=True` and accept edge weights. GIN requires an explicit MLP per layer and does not use edge weights since its expressiveness is encoded in the epsilon-based self-feature weighting. BatchNorm is not applied after GNN layers because the effective batch over which normalization occurs is only 12 nodes per sample (the number of leads), which is insufficient for stable batch statistics.

Lead importance scores are computed in `get_lead_importance()` as the L2 norm of each node's final GNN embedding, normalized across leads. Higher norm indicates greater contribution to the classification decision.

Edges are retained only where the hybrid weight exceeds `correlation_threshold`, pruning weak or noisy connections. This construction is particularly motivated by the localized nature of Brugada syndrome (typically expressed in the right precordial leads V1 and V2). This retrain edge weight only happen on GAT since GCN will used initial edge weight and GIN will not use edge weight.

#### Mathematical Formulation

The SpatialGNN processes each patient's 12-lead ECG through two sequential stages: temporal encoding and spatial aggregation.

**Temporal Encoding.** Each lead signal $x_i \in \mathbb{R}^{T}$ is independently passed through a shared per-lead ResNet encoder $f_\theta$ to produce a node feature vector:

$$h_i^{(0)} = f_\theta(x_i), \quad h_i^{(0)} \in \mathbb{R}^{D}$$

where $T = 1200$ (samples per lead) and $D$ is the embedding dimension controlled by `resnet_channels`.

**Graph Convolution.** Node features are updated over $L$ GNN layers. For a GCN layer, the update rule follows:

$$h_i^{(l+1)} = \sigma\!\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{w_{ij}}{\sqrt{d_i d_j}}\, W^{(l)} h_j^{(l)} \right)$$

where $\mathcal{N}(i)$ is the neighbourhood of node $i$ defined by the hybrid adjacency matrix, $w_{ij}$ is the edge weight, $d_i$ is the weighted degree of node $i$, $W^{(l)}$ is a learnable weight matrix, and $\sigma$ is ELU activation.

For GAT layers, the aggregation is instead governed by learned attention coefficients:

$$\alpha_{ij} = \frac{\exp\!\left(\text{LeakyReLU}(a^\top [W h_i^{(l)} \| W h_j^{(l)}])\right)}{\sum_{k \in \mathcal{N}(i)} \exp\!\left(\text{LeakyReLU}(a^\top [W h_i^{(l)} \| W h_k^{(l)}])\right)}$$

$$h_i^{(l+1)} = \sigma\!\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W h_j^{(l)} \right)$$

**Graph Readout.** After $L$ layers, global average pooling aggregates all node embeddings into a single graph-level representation:

$$z = \frac{1}{12} \sum_{i=1}^{12} h_i^{(L)}, \quad z \in \mathbb{R}^{D}$$

**Multi-Task Classification Head.** The graph embedding $z$ is passed to task-specific MLP heads. The total training objective is a weighted sum of focal losses across the primary task (Brugada classification) and auxiliary tasks:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FL}}^{\text{brugada}} + \lambda \sum_{k} \mathcal{L}_{\text{FL}}^{(k)}$$

where $\lambda = 0.3$ weights the auxiliary contributions. Focal loss for each task is defined as:

$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

with $\alpha = 0.79$ derived from the inverse class frequency $(287/363)$ and $\gamma = 2.0$.


## Training

### Single Run
To train a model with a specific config:

```bash
python main.py --config configs/resnet_baseline.yml
```

To override specific config values without editing the YAML:

```bash
python main.py --config configs/spatial_gnn.yml --override training.learning_rate=0.001 data.batch_size=32 data.augmentation.enabled=true
```

### Class Imbalance Handling
The dataset has approximately a 1:3.8 positive-to-negative ratio. Two mechanisms address this simultaneously:
1. `WeightedRandomSampler` oversamples the minority class during training so each batch contains a more balanced class distribution.
2. Focal loss with `alpha=0.79` (computed from the class frequency) and `gamma=2.0` downweights easy negatives and focuses gradient updates on hard examples and the minority class.

## Hyperparameter Search
Hyperparameter search is run per model using random sampling over a predefined search space. Each trial runs the model with three fixed seeds (42, 43, 44) and reports the mean and standard deviation of the primary metric across seeds.

```bash
python -m src.scripts.hyperparameter_search --model spatial_gnn --search random --n_random 20
```

## Model Comparison
After hyperparameter search is complete for all models, the final comparison evaluates each best config across five seeds (42 through 46):

```bash
python -m src.scripts.compare_models
```

This script reads from `configs/best/` for each model, runs training and evaluation per seed, aggregates mean and standard deviation of all metrics across seeds, and writes the final comparison table to `experiments/model_comparison.csv`.

## Interpretability

The interpretability pipeline covers all three model types, combining **Grad-CAM** for temporal attribution, **GNN Lead Importance** for spatial attribution, and **Permutation Importance** for classical feature attribution.

When models are evaluated, XAI reports are automatically generated for both a Brugada-positive (TP) patient and, where available, a False Negative (FN) patient — enabling direct comparison of what the model attends to when it succeeds versus when it misses a diagnosis.

The visual outputs per model type are:

**Spatial GNN**
- **Temporal Grad-CAM**: Heatmaps overlaying the raw ECG signal to pinpoint *when* the model detects pathological morphology (e.g., coved-type ST-segment elevation at the QRS-ST junction).
- **Lead Importance Bar Chart**: Derived from the L2 norm of graph node embeddings, showing which leads (e.g., V1, V2) drove the classification.
- **Lead Connectivity Graph**: Node-link diagram of the hybrid adjacency edge weights over the patient's specific cardiac electrical state.

**1D ResNet**
- **Temporal Grad-CAM**: Same heatmap overlay as above; activation patterns tend to be more diffuse compared to the GNN.

**HGB**
- **Permutation Feature Importance**: Global bar chart showing which handcrafted morphological features (ST elevation, J-point amplitude, T-wave polarity, etc.) most influenced the classifier. Features from Brugada-relevant leads V1–V3 are highlighted in red.

---
### HGB — Feature Importance (Patient 3072836, True Positive)
![HGB XAI](https://raw.githubusercontent.com/kiuyha/idsc2026-brugada/main/experiments/xai_hgb_baseline_3072836.png)

> **Reading this plot**: Each bar shows how much a feature's permutation (random shuffling) decreases model accuracy — higher means the feature is more critical. Bars in **red** belong to Brugada-relevant leads V1, V2, or V3. For this correctly classified patient, `j_point_V3`, `min_V2`, and `st_elevation_V3` dominate, which aligns with the clinical hallmark of right precordial ST-segment elevation in Brugada syndrome.

### HGB — Feature Importance (Patient 581912, False Negative)
![HGB FN XAI](https://raw.githubusercontent.com/kiuyha/idsc2026-brugada/main/experiments/xai_hgb_baseline_581912_fn.png)

> **Reading this plot**: For this missed Brugada case, the most important features shift toward `st_elevation_V2` and `st_elevation_aVL`, but the overall importance magnitudes are lower and QRS duration features show near-zero importance. This suggests the patient's morphological features were atypical or subtle enough that the handcrafted feature set failed to capture sufficient discriminative signal — a key limitation of HGB compared to learned representations.

### 1D ResNet — Grad-CAM (Patient 3072836, True Positive)
![ResNet XAI](https://raw.githubusercontent.com/kiuyha/idsc2026-brugada/main/experiments/xai_resnet_baseline_3072836.png)

> **Reading this plot**: Red regions indicate high Grad-CAM intensity — the temporal windows the model weighted most heavily. For this correctly classified patient, activations are broadly distributed across the full 12-second window with no clear anatomical focus, reflecting the ResNet's purely temporal processing with no spatial lead-awareness.

### 1D ResNet — Grad-CAM (Patient 3004284, False Negative)
![ResNet FN XAI](https://raw.githubusercontent.com/kiuyha/idsc2026-brugada/main/experiments/xai_resnet_baseline_3004284_fn.png)

> **Reading this plot**: For this missed Brugada case, activations remain diffuse and indistinguishable from the TP report above — the model attends to the same general regions regardless of outcome. This inability to localise discriminative morphology is consistent with the ResNet's black-box nature and explains why it fails silently on atypical presentations.

### Spatial GNN — Grad-CAM + Lead Importance (Patient 3072836)
![Spatial GNN XAI](https://raw.githubusercontent.com/kiuyha/idsc2026-brugada/main/experiments/xai_spatial_gnn_3072836.png)

> **Reading this plot**: Top three panels show Grad-CAM overlays for the highest-importance leads — high-intensity regions are focused at the early portion of each cardiac cycle, consistent with the QRS-ST junction where coved-type ST elevation manifests. Bottom-left shows lead importance scores derived from L2 norms of graph embeddings (V1 highest at 0.142, followed by aVL and V2), directly corroborating clinical diagnostic criteria. Bottom-right shows the lead connectivity graph where node size encodes importance and edge weight encodes hybrid adjacency strength — precordial leads form a tightly interconnected subgraph, confirming the GNN has learned anatomical proximity implicitly.

> **Note**: No FN report is generated for the Spatial GNN on this evaluation seed (seed=42), as the model achieved perfect recall (1.0) on the test split, all Brugada-positive patients were correctly identified.

---

The `notebook.ipynb` workspace provides an interactive environment to load specific patient IDs, run inference, and generate these visualizations post-hoc.
## Reproducibility
- All randomness in the pipeline is controlled through a single seed value in the config, which is passed to Python's `random`, NumPy, and PyTorch (including CUDA) via `set_seed()` in `utils.py`. CUDA determinism is enforced via `torch.backends.cudnn.deterministic = True`.

- The train/validation/test split is performed using stratified sampling with the config seed as the `random_state`, ensuring the same seed always produces the same split. This means that changing only the seed in the comparison script genuinely varies both the data partition and the weight initialization independently.

- To exactly reproduce the reported results, run `compare_models.py` using the configs in `configs/best/` without modification. The hyperparameter search results in `experiments/` document the full search history for each model.

## References
Goldberger, A. L., Goldberger, Z. D., & Shvilkin, A. (2013). Goldberger's Clinical Electrocardiography: A Simplified Approach (8th ed.). Elsevier Saunders. ISBN: 978-0-323-08786-5.

## Citation
If using the Brugada-HUCA dataset, cite:

```bibtex
@article{PhysioNet-brugada-huca-1.0.0,
  author = {{Costa Cortez}, Nahuel and {Garcia Iglesias}, Daniel},
  title = {{Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada Syndrome}},
  journal = {{PhysioNet}},
  year = {2026},
  month = feb,
  note = {Version 1.0.0},
  doi = {10.13026/0m2w-dy83},
  url = {https://doi.org/10.13026/0m2w-dy83}
}
```
