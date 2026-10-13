# CNN Image Classification: From Zero to Confident

A masterclass educational repository and guided curriculum teaching Convolutional Neural Networks (CNNs) from first principles to modern transfer learning and interpretability.

## Curriculum Roadmap

1. **Step 1: Theory and Intuition** (`01_cnn_theory_and_intuition.ipynb`)
   - Why fully connected networks fail on image data (parameter counting)
   - Step-by-step 2D convolution worked by hand on a 5x5 image with a 3x3 filter
   - Stride, padding (valid vs. same), channels, feature maps, pooling
   - Core inductive biases: weight sharing, spatial locality, translation invariance, receptive field
2. **Step 2: 2D Convolution from Scratch** (`02_convolution_from_scratch.ipynb`)
   - Forward pass 2D convolution implemented in pure NumPy
   - Exact numerical verification against `torch.nn.functional.conv2d`
   - Visualizing classical image processing filters (Sobel horizontal/vertical, Laplacian, Gaussian blur)
3. **Step 3: Part A — CIFAR-10 Scratch CNN** (`03_cifar10_scratch_cnn.ipynb`)
   - Building a modular CNN architecture in PyTorch
   - Training pipeline with data augmentation and real-time metric tracking
   - Inspecting learned first-layer filters, layer-by-layer feature maps, confusion matrix & error analysis
4. **Step 4: Modern Architectures — VGG & ResNet** (`04_vgg_and_resnet_architectures.ipynb`)
   - Architectural evolution: deep 3x3 convolutions (VGG) vs. residual learning (ResNet)
   - Implementing residual blocks with identity and projection shortcuts
   - Empirical proof of the degradation problem (Plain deep net vs. ResNet)
5. **Step 5: Part B — Transfer Learning with ResNet-18** (`05_transfer_learning_resnet18.ipynb`)
   - Pretrained representations on a custom small dataset
   - Feature extraction (frozen backbone) vs. Fine-tuning (discriminative learning rates)
   - Systematic benchmark: Scratch training vs. Transfer learning
6. **Step 6: Interpretability with Grad-CAM** (`06_interpretability_gradcam.ipynb`)
   - Gradient-weighted Class Activation Mapping implemented from scratch with hooks
   - Heatmap visual grounding and model explainability
7. **Step 7: Real-World Failure Modes** (`07_failure_demos.ipynb`)
   - Failure Mode A: Texture bias & frequency reliance
   - Failure Mode B: Adversarial vulnerability with Fast Gradient Sign Method (FGSM)
   - Failure Mode C: Domain shift degradation and mitigation strategies
8. **Step 8: "Prove It" Assessment & Coding Exercises** (`08_exercises_and_prove_it.ipynb`)
   - 8 progressive comprehension questions
   - 3 guided coding challenges with unit test assertions
   - 1 open-ended research extension project

---

## Hardware Requirements & Batch Size Guidance
- **CPU**: Supported across all notebooks. Recommended batch sizes: 32 - 64.
- **GPU**: Optimized for CUDA GPUs (including 6GB - 8GB VRAM like NVIDIA RTX 4060). Recommended batch sizes: 64 - 128.
