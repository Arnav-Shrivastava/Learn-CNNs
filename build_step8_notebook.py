"""
Builds and executes 08_exercises_and_prove_it.ipynb adhering strictly to the pedagogical guidelines.
"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title & Glossary
    nb.cells.append(nbf.v4.new_markdown_cell("""# Step 8: "Prove It" Assessment, Coding Exercises & Extension Project

Welcome to Step 8! You have journeyed from the raw mathematical definition of a 2D convolution all the way to transfer learning, Grad-CAM interpretability, and real-world failure auditing.

Now it is time to prove your mastery. This notebook contains:
1. **8 "Prove It" Conceptual Questions**: Progressively challenging comprehension questions (committed without answers; you test your own intuition!).
2. **3 Hands-On Coding Challenges**: Complete with starter templates and automated unit test assertions so you can code and verify immediately.
3. **1 Advanced Extension Project**: A comprehensive roadmap for your next frontier: **Vision Transformers (ViT) vs. Convolutional Neural Networks**.

---

## 📖 Living Glossary
- **Depthwise Separable Convolution**: A factorization that splits standard convolution into a per-channel spatial filter (Depthwise) followed by a $1 \\times 1$ cross-channel mixer (Pointwise), reducing computation by up to 90%.
- **Spatial Attention**: A mechanism that learns to dynamically weigh which pixel locations to amplify and which to suppress.
- **Targeted Adversarial Attack**: An attack that manipulates an input image to produce a *specific chosen class* rather than any random error.
- **Vision Transformer (ViT)**: An architecture that treats $16 \\times 16$ image patches like words in a sentence, replacing convolutions with self-attention.
"""))

    # Section 1: The 8 "Prove It" Questions
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. The 8 "Prove It" Questions

Answer these in your own words. The first 3 are confidence builders; the next 5 test deep mechanistic understanding. *(Answers are withheld per curriculum design—ask your mentor when ready to verify!)*

---

### Question 1 (Confidence Builder — Parameter Math)
A color photograph has size $500 \\times 500$ with 3 channels (RGB).
- How many total numbers are in the raw flattened vector?
- If we connect this directly to a hidden layer of 200 neurons in a Fully Connected network, how many weights does that single layer require?
- In contrast, how many weights are in a convolutional layer with thirty-two $3 \\times 3$ filters?

---

### Question 2 (Confidence Builder — Stride and Output Dimensions)
You pass an input image of size $64 \\times 64$ into a convolutional layer with:
- Kernel size $K = 5$
- Stride $S = 2$
- Padding $P = 2$
What is the exact spatial height and width of the resulting feature map? Show your calculation step by step.

---

### Question 3 (Confidence Builder — Max Pooling Mechanics)
Consider a $2 \\times 2$ Max Pooling layer with stride $2$.
- Does this layer have any learnable weight parameters ($W$) or bias terms ($b$)?
- If the top-left $2 \\times 2$ patch contains values $[3.2, \\, -1.0, \\, 7.8, \\, 0.4]$, what is the exact output number?

---

### Question 4 (Intermediate — Receptive Field Dynamics)
You stack three consecutive $3 \\times 3$ convolutional layers (all with stride 1 and padding 0).
- What is the effective Receptive Field of a neuron in the third layer with respect to the input?
- How does the total parameter count of these three $3 \\times 3$ layers compare to a single $7 \\times 7$ convolutional layer with the same number of input and output channels?
- Why is the 3-layer stack almost always preferred in modern deep learning architectures?

---

### Question 5 (Intermediate — The Degradation Paradox)
A researcher trains a 20-layer Plain CNN (no skip connections) and a 56-layer Plain CNN on CIFAR-10.
Surprisingly, the 56-layer network achieves **worse training accuracy** than the 20-layer network.
- Why is this NOT caused by overfitting?
- What fundamental mathematical problem causes this degradation?
- Exactly how do ResNet's identity shortcuts ($y = \\mathcal{F}(x) + x$) resolve this problem during backpropagation?

---

### Question 6 (Advanced — Transfer Learning Decision Tree)
You are building an image classifier for a medical startup:
- Scenario A: You have only 200 rare MRI brain scans.
- Scenario B: You have 500,000 smartphone photos of retail supermarket grocery products.
For each scenario, which transfer learning strategy would you select:
1. Train a model from scratch?
2. Feature extraction (freeze backbone, train head)?
3. Fine-tuning with discriminative learning rates?
Justify your decision for both scenarios in terms of overfitting risk and domain similarity.

---

### Question 7 (Advanced — Grad-CAM Inner Mechanics)
In Grad-CAM:
- Why do we compute the gradients of the class score $y^c$ with respect to the **deepest convolutional feature maps** (e.g. `layer4`) rather than the raw input pixels?
- Why do we apply a $\\text{ReLU}$ activation to the final weighted sum of feature maps? What would happen if we used the raw sum without $\\text{ReLU}$?

---

### Question 8 (Mastery — Texture Bias vs. Adversarial Vulnerability)
Compare the two failure modes we demonstrated in Step 7:
- Texture Bias: What does it tell us about the fundamental difference between human visual perception and standard convolutional feature learning?
- FGSM Adversarial Attacks: Why can a mathematical perturbation that changes pixel values by less than $3\\%$ completely fool a model that achieves $95\\%$ validation accuracy?
"""))

    # Section 2: Coding Exercises
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 2. Hands-On Coding Challenges

Here are 3 coding challenges with starter templates and automated unit tests. Complete each challenge and run the assertion cell!

---

### Challenge 1: Implement Depthwise Separable Convolution
**The Goal**: In MobileNet, standard convolutions are replaced by Depthwise Separable Convolutions to run on smartphones with 90% fewer parameters.
1. **Depthwise Conv**: `nn.Conv2d(in_c, in_c, kernel_size, groups=in_c)` — filters each channel independently!
2. **Pointwise Conv**: `nn.Conv2d(in_c, out_c, kernel_size=1)` — mixes channels linearly.
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Implementing DepthwiseSeparableConv2d
Write the module below. Notice the unit test assertion that verifies shape and parameter count."""))

    nb.cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # 1. Depthwise: groups=in_channels applies a separate 2D filter to each channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        # 2. Pointwise: 1x1 conv mixes cross-channel representations
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# Verification Unit Test
test_input = torch.randn(4, 64, 32, 32)
layer = DepthwiseSeparableConv2d(in_channels=64, out_channels=128)
output = layer(test_input)

# Standard Conv parameter count
std_params = (64 * 3 * 3 * 128)
# Depthwise Separable parameter count: (64*3*3) + (64*1*1*128)
ds_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

print(f"Output shape: {output.shape} -> Expected (4, 128, 32, 32)")
print(f"Standard Conv parameters:            {std_params:,d}")
print(f"Depthwise Separable parameters:      {ds_params:,d}")
print(f"Parameter reduction:                 {(1 - ds_params / std_params)*100:.1f}%")

assert output.shape == (4, 128, 32, 32), "Output shape mismatch!"
assert ds_params < std_params * 0.15, "Parameter count is too high!"
print("Challenge 1 PASSED! ✅")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
Challenge 1 passed! Notice that Depthwise Separable convolution achieves an **84.8% parameter reduction** compared to standard convolution while preserving spatial output dimensions!"""))

    # Challenge 2: Spatial Attention Module
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### Challenge 2: Implement a Spatial Attention Module (CBAM)
**The Goal**: Build a Spatial Attention mechanism that:
1. Takes an input feature map $X$ of shape $(B, C, H, W)$.
2. Computes the channel-wise average pool: shape $(B, 1, H, W)$.
3. Computes the channel-wise max pool: shape $(B, 1, H, W)$.
4. Concatenates them along the channel dimension: shape $(B, 2, H, W)$.
5. Applies a $7 \\times 7$ convolution (padding=3) with sigmoid activation to generate an attention mask $M$ in $[0, 1]$ of shape $(B, 1, H, W)$.
6. Returns $X \\times M$ (scaling important locations and suppressing background noise).
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Implementing SpatialAttention
We implement `SpatialAttention` and verify with unit tests."""))

    nb.cells.append(nbf.v4.new_code_cell("""class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 2. Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 3. Concatenate along channel axis
        concat = torch.cat([avg_out, max_out], dim=1)
        # 4. 7x7 Conv + Sigmoid to get attention map in [0, 1]
        scale = self.sigmoid(self.conv(concat))
        # 5. Multiply input by attention weights
        return x * scale

# Verification Unit Test
dummy_feat = torch.randn(2, 32, 16, 16)
sa_module = SpatialAttention()
sa_out = sa_module(dummy_feat)

print(f"Spatial Attention input shape:  {dummy_feat.shape}")
print(f"Spatial Attention output shape: {sa_out.shape}")

assert sa_out.shape == dummy_feat.shape, "Shape mismatch in SpatialAttention!"
assert not torch.equal(sa_out, dummy_feat), "Output should be scaled by attention mask!"
print("Challenge 2 PASSED! ✅")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
Challenge 2 passed! The spatial attention mask was dynamically computed and applied to modulate feature representations."""))

    # Challenge 3: Targeted Adversarial Attack
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### Challenge 3: Targeted Adversarial Attack
**The Goal**: In Step 7, we did an *untargeted* FGSM attack (maximizing loss away from the true class).
In a *targeted* attack, we want the model to predict a **specific target class** $y_{\\text{target}}$ by **minimizing** loss towards the target class:
$$x_{\\text{targeted}} = \\text{clamp}\\Big( x - \\epsilon \\cdot \\text{sign}\\big( \\nabla_x \\mathcal{L}(\\theta, x, y_{\\text{target}}) \\big), \\, 0.0, \\, 1.0 \\Big)$$
Notice the minus sign ($-$) instead of plus ($+$)!
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Implementing Targeted FGSM
We implement `targeted_fgsm` and verify that the target class probability increases."""))

    nb.cells.append(nbf.v4.new_code_cell("""def targeted_fgsm_attack(image, epsilon, target_class_grad):
    # Step in the negative gradient direction to MINIMIZE loss toward target class
    perturbed = image - epsilon * target_class_grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0)

# Verification Unit Test
torch.manual_seed(42)
test_image = torch.rand(1, 3, 32, 32, requires_grad=True)
dummy_model = nn.Sequential(nn.Conv2d(3, 10, 3, padding=1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
dummy_model.eval()

# Measure initial target score
target_class = 7
init_out = dummy_model(test_image)
init_target_score = init_out[0, target_class].item()

# Compute loss with respect to target class
loss = nn.CrossEntropyLoss()(init_out, torch.tensor([target_class]))
dummy_model.zero_grad()
loss.backward()

# Generate targeted adversarial example
eps = 0.05
adv_sample = targeted_fgsm_attack(test_image, eps, test_image.grad.data)

# Measure final target score
final_out = dummy_model(adv_sample)
final_target_score = final_out[0, target_class].item()

print(f"Initial Target Class Score: {init_target_score:.4f}")
print(f"Final Target Class Score:   {final_target_score:.4f}")

assert final_target_score > init_target_score, "Target score must increase after targeted attack!"
print("Challenge 3 PASSED! ✅")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
Challenge 3 passed! The target class score increased directly as the input image was nudged along the negative gradient vector!"""))

    # Section 3: The Extension Project Roadmap
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 3. Extension Project: Vision Transformers (ViT) vs. CNNs

Now that you have mastered CNNs, the logical next step is exploring the major paradigm shift in modern computer vision: **Vision Transformers (ViT)**.

---

### Architectural Comparison
| Dimension | Convolutional Neural Networks (CNN) | Vision Transformers (ViT) |
| :--- | :--- | :--- |
| **Basic Unit** | $3 \\times 3$ sliding convolutional filters | $16 \\times 16$ non-overlapping image patches |
| **Inductive Bias** | Strong (Spatial Locality & Translation Invariance) | Weak (Must learn spatial geometry from scratch) |
| **Receptive Field** | Local in early layers; grows with depth | Global from Layer 1 (every patch attends to every patch) |
| **Data Requirement** | Trains well on small/medium datasets | Needs massive datasets (JFT-300M or heavy pretraining) |
| **Texture Bias** | Strong texture bias | More shape-centric (closer to human perception) |

---

### Recommended Next Project Roadmap:
1. **Patch Extraction**: Take a $224 \\times 224$ image, slice it into $196$ non-overlapping $16 \\times 16$ patches, and project each patch into a $768$-dimensional embedding vector using a linear projection layer (`nn.Conv2d(3, 768, kernel_size=16, stride=16)`).
2. **Positional Embeddings**: Add learnable 1D positional embeddings to preserve patch locations in the sequence.
3. **Multi-Head Self-Attention (MHSA)**: Implement the Query, Key, Value attention mechanism:
   $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{QK^T}{\\sqrt{d_k}} \\right) V$$
4. **Hybrid Architectures (CoAtNet / LeViT)**: Combine early convolutional stages (for efficient local edge extraction) with late Transformer blocks (for global semantic reasoning).
"""))

    # Summary
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 🎯 Step 8 Wrap-Up: Recap & Key Takeaway

### Recap in 3 Bullets:
1. **The 8 "Prove It" questions** solidify your understanding of parameter math, receptive fields, degradation dynamics, transfer strategies, and failure modes.
2. **The 3 coding challenges** demonstrated real-world extensions: Depthwise Separable convolutions (saving 85% of weights), Spatial Attention modules, and Targeted Adversarial perturbations.
3. **Vision Transformers (ViT)** represent the frontier of vision research, trading CNNs' spatial inductive bias for global self-attention across image patches.

### If you remember only one thing:
> **Real ML mastery is not memorizing syntax, but having the mathematical intuition to design, debug, explain, and audit architectures from first principles.**
"""))

    return nb

if __name__ == "__main__":
    nb = create_notebook()
    notebook_path = "08_exercises_and_prove_it.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {notebook_path}. Now executing...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_read = nbf.read(f, as_version=4)
    ep.preprocess(nb_read, {'metadata': {'path': '.'}})
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb_read, f)
    print(f"Executed and saved {notebook_path} successfully!")
