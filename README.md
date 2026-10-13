# CNN Image Classification: From Zero to Confident

A masterclass educational curriculum and runnable repository teaching Convolutional Neural Networks (CNNs) from first principles to modern transfer learning, interpretability, and real-world failure auditing.

Developed with a beginner-first pedagogical approach: short intuitive explanations, real-world analogies, step-by-step arithmetic worked by hand, tensor shape tracing at every layer, and robust PyTorch implementations.

---

## 🌟 Visual Showcase

| Feature / Artifact | Visualization | Description |
| :--- | :--- | :--- |
| **Grad-CAM Grounding** | ![Grad-CAM](assets/gradcam_visualization.png) | Visualizing class-discriminative attention heatmaps from ResNet-18 layer4 activations. |
| **Adversarial Vulnerability** | ![Adversarial FGSM](assets/adversarial_attack_fgsm.png) | Generating imperceptible Fast Gradient Sign Method ($\epsilon=0.03$) noise that flips predictions. |
| **Learned Conv Filters** | ![Learned Filters](assets/cifar10_learned_filters.png) | 32 first-layer filters autonomously discovering color gradients and directional edges. |
| **Layer Feature Maps** | ![Feature Maps](assets/cifar10_feature_maps.png) | Layer-by-layer representations evolving from geometric contours to abstract semantics. |
| **Degradation Problem** | ![Degradation](assets/degradation_problem_comparison.png) | Empirical proof showing deep Plain nets stalling while ResNet residual shortcuts converge. |
| **Transfer Benchmark** | ![Transfer Learning](assets/transfer_learning_comparison.png) | Benchmark comparing Scratch vs. Feature Extraction vs. Fine-Tuning on small data. |
| **Texture Bias Audit** | ![Texture Bias](assets/texture_bias_demo.png) | Showing model confidence collapse when surface textures are removed. |
| **Domain Shift Collapse**| ![Domain Shift](assets/domain_shift_demo.png) | Quantifying accuracy drops when test images shift to night and sketch domains. |

---

## 📊 Benchmark Results: Scratch vs. Transfer Learning (500 Images)

Empirical evaluation on a challenging 5-class custom small-data benchmark (100 training images per class, 50 test images per class):

| Strategy | Backbone Status | Trainable Parameters | Best Test Accuracy | Optimization Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **1. Scratch (No Pretraining)** | Randomly Initialized | 11,179,077 (100%) | **~48.0%** | Severely overfits; struggles to generalize with 500 samples. |
| **2. Feature Extraction** | Fully Frozen | 2,565 (0.02%) | **~90.4%** | Fast convergence in epoch 1; zero risk of forgetting. |
| **3. Fine-Tuning (Discriminative)**| `layer4` + `fc` Unfrozen | 8,397,829 (75%) | **~94.8%** | Top accuracy; adapts semantic features with $\eta_{\text{low}}=10^{-4}$. |

---

## 📚 Curriculum Roadmap

The project is structured into 8 modular, standalone, fully executed Jupyter Notebooks:

### 1. Theory & Intuition (`01_cnn_theory_and_intuition.ipynb`)
- Why fully connected networks fail on images (parameter explosion: billions of weights for smartphone photos).
- Core inductive biases: **Spatial Locality** and **Weight Sharing**.
- Step-by-step arithmetic: 2D convolution worked cell-by-cell by hand on a $5 \times 5$ image with a $3 \times 3$ filter.
- Stride, "Valid" vs. "Same" padding formulas, multi-channel tensors $(B, C, H, W)$.
- Max vs. Average pooling and mathematical Receptive Field growth formulas.

### 2. 2D Convolution from Scratch (`02_convolution_from_scratch.ipynb`)
- Built a forward-pass 2D convolution engine in pure NumPy supporting arbitrary stride and padding.
- Numerically certified output against PyTorch's native `torch.nn.functional.conv2d` (tolerance $< 10^{-7}$).
- Visualized classical image processing filters (Sobel horizontal/vertical, Laplacian, Gaussian blur, Sharpen) on a high-resolution photograph.

### 3. Part A — CIFAR-10 Scratch CNN (`03_cifar10_scratch_cnn.ipynb`)
- Built `SmallCIFARNet` in PyTorch with Batch Normalization, Dropout, and Max Pooling.
- Full training pipeline with data augmentation (`RandomCrop`, `RandomHorizontalFlip`).
- Plotted loss and accuracy curves reaching ~80% test accuracy.
- Extracted and visualized the 32 learned first-layer RGB filter weights.
- Traced layer-by-layer feature map activations across all three network depths.
- Generated a normalized $10 \times 10$ Confusion Matrix and inspected misclassified test examples.

### 4. Modern Architectures — VGG & ResNet (`04_vgg_and_resnet_architectures.ipynb`)
- Architectural theory: why stacking two $3 \times 3$ filters saves 28% of parameters over a single $5 \times 5$ filter while adding non-linear expressiveness.
- The **Degradation Problem**: why deeper plain networks suffer from optimization failure (not overfitting).
- Implemented `ResidualBlock` with identity and projection shortcuts.
- Head-to-head empirical experiment: 17-layer Plain CNN vs. 17-layer ResNet, proving residual shortcuts accelerate convergence.

### 5. Part B — Transfer Learning with ResNet-18 (`05_transfer_learning_resnet18.ipynb`)
- Pretrained representations on a small 5-class custom dataset (500 training images).
- Systematically compared Scratch vs. Feature Extraction (frozen backbone) vs. Fine-Tuning.
- Implemented Discriminative Learning Rates ($\eta=10^{-4}$ for `layer4`, $\eta=10^{-3}$ for classification head).

### 6. Interpretability & Explainable AI — Grad-CAM (`06_interpretability_gradcam.ipynb`)
- Implemented Gradient-weighted Class Activation Mapping from scratch using PyTorch hooks.
- Extracted gradients with respect to `layer4` feature maps, performed Global Average Pooling, and calculated ReLU-activated class heatmaps.
- Generated alpha-blended overlays showing model visual grounding and counter-factual class queries.

### 7. Real-World Failure Modes (`07_failure_demos.ipynb`)
- **Texture Bias**: Evaluated clean vs. blurred vs. contour-only images, proving CNNs rely on surface texture rather than 3D shape.
- **Adversarial Vulnerability**: Implemented the Fast Gradient Sign Method (FGSM) from scratch ($\epsilon=0.03$), demonstrating prediction flips on human-imperceptible noise.
- **Domain Shift**: Measured quantitative performance collapse on simulated night and sketch domains, and discussed mitigation strategies (domain randomization, AdaIN, self-supervised pretraining).

### 8. "Prove It" Assessment & Coding Exercises (`08_exercises_and_prove_it.ipynb`)
- 8 progressive comprehension questions (committed without answers to test conceptual mastery).
- 3 guided coding challenges with automated unit test assertions:
  1. Depthwise Separable Convolutions (MobileNet factorized convs).
  2. Spatial Attention Module (CBAM-style channel pooling and $7 \times 7$ attention mask).
  3. Targeted Adversarial Attack (minimizing loss toward a chosen target class).
- 1 open-ended research extension project: Vision Transformers (ViT) vs. CNNs.

---

## 🛠️ Hardware Requirements & Batch Size Guidance

This repository is designed to execute seamlessly on both CPU and GPU hardware:
- **NVIDIA GPU (CUDA)**: Tested on an 8GB VRAM NVIDIA GeForce RTX 4060 Laptop GPU.
  - Recommended batch sizes: `64` to `128` (CIFAR-10) and `32` (ResNet-18 $224 \times 224$).
  - Full training takes ~1 to 2 minutes per notebook.
- **CPU**: Supported natively across all notebooks with automatic device fallback.
  - Recommended batch sizes: `32` to `64`.
  - Representative subsets are configured to ensure execution finishes within ~5 minutes.

---

## 🚀 How to Run Locally

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/Arnav-Shrivastava/Learn-CNNs.git
cd Learn-CNNs

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
The CIFAR-10 dataset will automatically download on first run. If you prefer to download it manually:
```bash
python -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', train=True, download=True)"
```
*Note: Datasets and model checkpoints are excluded from Git via `.gitignore`.*

### 3. Launch Notebooks
```bash
jupyter notebook
```
Open notebooks in numbered order `01` through `08`. Every cell has been pre-executed with saved outputs, interactive plots, and printed tensor shapes.

---

## 📂 Repository Structure

```
Learn-CNNs/
├── .gitignore                             # Git ignore rules for data, weights, checkpoints
├── requirements.txt                       # Project package dependencies
├── README.md                              # Comprehensive project documentation
├── assets/                                # Embedded visualization figures
│   ├── cat_sample.jpg                     # High-resolution benchmark test photograph
│   ├── classic_filters_visualization.png  # Sobel, Laplacian, Gaussian blur comparisons
│   ├── cifar10_training_curves.png        # Training & validation loss/accuracy curves
│   ├── cifar10_learned_filters.png        # 32 first-layer convolutional RGB filters
│   ├── cifar10_feature_maps.png           # Layer-by-layer intermediate activation maps
│   ├── cifar10_confusion_matrix.png       # Normalized 10x10 confusion matrix heatmap
│   ├── cifar10_misclassified.png          # Misclassified test examples error analysis
│   ├── degradation_problem_comparison.png # Plain-17 vs. ResNet-17 degradation benchmark
│   ├── transfer_learning_comparison.png   # Scratch vs. Frozen vs. Fine-tuning curves
│   ├── gradcam_visualization.png          # Grad-CAM heatmap & attention overlays
│   ├── texture_bias_demo.png              # Texture degradation vs. classification confidence
│   ├── adversarial_attack_fgsm.png        # FGSM clean vs. perturbation vs. adversarial
│   └── domain_shift_demo.png              # In-distribution vs. Night vs. Sketch shift
├── 01_cnn_theory_and_intuition.ipynb      # Step 1: Hand-worked convolution & theory
├── 02_convolution_from_scratch.ipynb      # Step 2: NumPy conv engine & PyTorch validation
├── 03_cifar10_scratch_cnn.ipynb           # Step 3: Part A CIFAR-10 scratch training
├── 04_vgg_and_resnet_architectures.ipynb  # Step 4: VGG, ResNet blocks & degradation
├── 05_transfer_learning_resnet18.ipynb    # Step 5: Part B Transfer learning benchmark
├── 06_interpretability_gradcam.ipynb      # Step 6: Grad-CAM from scratch with PyTorch hooks
├── 07_failure_demos.ipynb                 # Step 7: Texture bias, FGSM attack & domain shift
└── 08_exercises_and_prove_it.ipynb        # Step 8: Assessment questions & coding challenges
```

---

## 💡 What I Learned

1. **Why Convolutions Work**: Convolutions preserve 2D topology and use weight sharing to dramatically shrink parameters compared to dense networks.
2. **Receptive Field Geometry**: Stacking small $3 \times 3$ filters yields identical receptive fields to large filters with fewer weights and more non-linear expressiveness.
3. **The Power of Residuals**: Skip connections provide a direct gradient highway ($\frac{\partial \mathcal{L}}{\partial x}$ contains a $+1$ term), eliminating vanishing gradients and solving the degradation problem.
4. **Transfer Learning Dominance**: Pretrained backbones contain rich, generalizable visual primitives; on small datasets, fine-tuning reaches $>94\%$ accuracy where scratch training fails.
5. **Auditing Beyond Accuracy**: High benchmark accuracy can hide texture bias, adversarial vulnerability, and domain sensitivity; explainability tools like Grad-CAM are essential for trustworthy deployment.
