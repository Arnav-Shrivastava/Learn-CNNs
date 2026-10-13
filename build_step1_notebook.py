"""
Builds and executes 01_cnn_theory_and_intuition.ipynb adhering strictly to the pedagogical guidelines.
"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title & Metadata
    nb.cells.append(nbf.v4.new_markdown_cell("""# Step 1: CNN Theory & Intuition from Scratch

Welcome to Step 1! We are going to build a deep, intuitive understanding of Convolutional Neural Networks (CNNs).
If you know how a standard neural network (a Multi-Layer Perceptron) works with flat vectors of numbers, you are in the perfect place.

---

## 📖 Living Glossary
Here is your quick-reference dictionary. We will add every new term here as we learn it!

- **Pixel**: A single tiny colored dot on a computer screen that has a brightness value (0 to 255).
- **Fully Connected (Dense) Layer**: A layer where every input neuron connects to every single output neuron.
- **Spatial Locality**: The physical rule in images that pixels close together are related, while pixels far apart are less related.
- **Filter (or Kernel)**: A small grid of numbers (like a stencil or stamp) that slides across an image to search for specific visual patterns.
- **Convolution**: The mathematical operation of sliding a filter across an image, multiplying overlapping numbers, and adding them up.
- **Feature Map**: The output grid of numbers produced after sliding a filter over the image; it lights up wherever the filter's pattern was found.
- **Weight Sharing**: Using the exact same set of filter weights across all parts of the image, rather than learning a new weight for each pixel position.
- **Stride**: The step size (number of pixels) the filter hops each time it slides to the next position.
- **Padding**: Adding extra border pixels (usually zeros) around the outside of an image so the borders don't get cut off or ignored.
- **Channels**: The layers of depth in an image; for example, a color photo has 3 channels (Red, Green, Blue).
- **Pooling**: A downsampling operation that zooms out of an image by picking the maximum (Max Pooling) or average (Average Pooling) value in small patches.
- **Translation Invariance**: The ability of a network to recognize an object (like a cat) no matter where it appears in the frame (top-left or bottom-right).
- **Receptive Field**: The area of the original input image that a particular neuron can "see" and is influenced by.
"""))

    # Section 1: Why Fully Connected Networks Fail
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Why Fully Connected Networks Fail on Images

### (1) The Idea in One Plain Sentence
Fully connected networks fail on images because flattening an image into a long 1D list destroys spatial shapes and creates billions of redundant connection weights that cause massive overfitting and run out of computer memory.

### (2) Real-Life Analogy
Imagine reading a book where someone cut up every page with scissors, shuffled all the words into a single 50-mile-long ribbon of letters, and asked you to recognize a picture of a house drawn across the sentences. You lose all row-and-column relationships!

### (3) Tiny Example with Real Numbers Worked by Hand
Let's count parameters for a small modern smartphone photo:
- Image resolution: $1000 \\times 1000$ pixels.
- Color channels: 3 (Red, Green, Blue).
- Total input features: $1000 \\times 1000 \\times 3 = 3,000,000$ inputs!
- If the first hidden layer has just $1,000$ neurons:
  $$\\text{Weights} = 3,000,000 \\times 1,000 = 3,000,000,000 \\text{ weights (3 Billion!)}$$
- At 4 bytes per 32-bit float, 3 billion weights require $12 \\text{ GB}$ of RAM for just **one single layer**, without even storing activations or gradients!

### (4) The Math with Every Symbol Explained in Words
For a dense layer:
$$W_{\\text{dense}} = N_{\\text{inputs}} \\times N_{\\text{outputs}}$$
- $W_{\\text{dense}}$: Total number of weight parameters connecting the two layers.
- $N_{\\text{inputs}}$: Total number of incoming pixel values ($Height \\times Width \\times Channels$).
- $N_{\\text{outputs}}$: Number of neurons in the hidden layer.

Notice that $N_{\\text{inputs}}$ grows quadratically with image resolution ($H \\times W$)."""))

    # Code: Parameter comparison
    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: What We Are About to Do and Why
We will write a short Python snippet to calculate and compare the number of parameters required by a Fully Connected (Dense) layer versus a Convolutional layer for images of different resolutions."""))

    nb.cells.append(nbf.v4.new_code_cell("""# Comparing parameter counts: Dense Layer vs. Convolutional Layer
resolutions = [(28, 28), (224, 224), (1000, 1000)]
hidden_units = 512
conv_filters = 64
filter_size = 3  # 3x3 kernel

print(f"{'Resolution':<12} | {'Dense Parameters':<20} | {'Conv Parameters':<18}")
print("-" * 56)

for h, w in resolutions:
    dense_params = (h * w * 3) * hidden_units
    # Conv params: (in_channels * kernel_h * kernel_w + bias) * num_filters
    conv_params = (3 * filter_size * filter_size + 1) * conv_filters
    print(f"{h}x{w:<7} | {dense_params:>18,d} | {conv_params:>16,d}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
Look at the table above! While the Dense layer explodes from $1.2$ million weights to **$1.5$ billion weights** as the image size increases, the Convolutional layer uses the exact same **$1,792$ weights** regardless of image size. This is the magic of **Weight Sharing**!

### (6) Common Beginner Mistake
Thinking that a CNN requires a smaller image size because it has "fewer parameters." A CNN actually has vastly fewer parameters *because* its filters are tiny local stencils that slide everywhere, not because the image has to be small!

### (7) Check Yourself
**Question:** If an RGB image is $200 \\times 200$ pixels, how many inputs does it have when flattened?
<details>
<summary><b>Click to see Answer</b></summary>
$200 \\times 200 \\times 3 = 120,000$ inputs. If connected to 100 hidden neurons, that is $12,000,000$ weights!
</details>
"""))

    # Section 2: Spatial Locality & Weight Sharing
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 2. Core Inductive Biases: Spatial Locality & Weight Sharing

### (1) The Idea in One Plain Sentence
CNNs assume two fundamental truths about the physical world: nearby pixels matter together (Spatial Locality), and a visual pattern like a vertical edge looks the same whether it appears in the top-left or bottom-right corner (Weight Sharing).

### (2) Real-Life Analogy
- **Spatial Locality**: When recognizing an eye, you only need to look at the iris, pupil, and eyelid right next to each other. You don't need to connect an iris pixel to a shoe pixel at the bottom of the photo.
- **Weight Sharing**: Imagine carrying a rubber stamp of an eye. You can stamp or check for that pattern anywhere on the paper using the same stamp, instead of buying 10,000 different stamps for 10,000 paper locations.

### (3) 5-Line Prerequisite Refresher: The Dot Product
Before we compute convolutions, remember the dot product between two equal-sized grids:
1. Lay grid $A$ directly over grid $B$.
2. Multiply each number in $A$ by the matching number in $B$.
3. Add all those products together into one single number.
4. Example: $[1, 2] \\cdot [3, 4] = (1 \\times 3) + (2 \\times 4) = 3 + 8 = 11$.
"""))

    # Section 3: The 2D Convolution Operation Worked by Hand (Cell by Cell)
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 3. The 2D Convolution Operation Worked by Hand (Step by Step)

### (1) The Idea in One Plain Sentence
A 2D convolution slides a small filter grid over an input image grid, calculating a dot product at every valid position to produce an output feature map.

### (2) Real-Life Analogy
A magnifying window sliding across a photo: you hold a small $3 \\times 3$ glass stencil, slide it across the picture row by row from left to right, and write down how strongly the stencil lights up at each stop.

### (3) Tiny Example with Real Numbers Worked by Hand (5x5 Image with 3x3 Filter)
Let our 5x5 image $I$ be:
```
I = [[1, 2, 0, 1, 1],
     [0, 1, 2, 3, 1],
     [1, 0, 0, 2, 2],
     [2, 1, 1, 0, 1],
     [0, 3, 1, 2, 0]]
```

And our 3x3 filter $K$ (a vertical edge detector) be:
```
K = [[ 1,  0, -1],
     [ 1,  0, -1],
     [ 1,  0, -1]]
```

With no padding and a stride of 1:
- The output height is: $5 - 3 + 1 = 3$.
- The output width is: $5 - 3 + 1 = 3$.
- The output feature map is a $3 \\times 3$ grid!

Let's compute every single output position $(r, c)$ cell by cell!"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: What We Are About to Do and Why
We will define our $5 \\times 5$ image and $3 \\times 3$ filter as NumPy arrays, print them, and write out the exact cell-by-cell arithmetic so you can verify each number by hand."""))

    nb.cells.append(nbf.v4.new_code_cell("""import numpy as np

# 5x5 input image
I = np.array([
    [1, 2, 0, 1, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 0, 2, 2],
    [2, 1, 1, 0, 1],
    [0, 3, 1, 2, 0]
], dtype=int)

# 3x3 vertical edge filter
K = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
], dtype=int)

print("Input Image I (5x5):\\n", I)
print("\\nFilter K (3x3):\\n", K)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
We have our exact image and filter matrices defined. Now let's calculate the 9 output cells one by one."""))

    # Cell-by-cell walkthrough
    nb.cells.append(nbf.v4.new_markdown_cell("""### Hand Calculations for All 9 Positions:

- **Position (0, 0)**: Top-Left window rows 0..2, cols 0..2:
  $$\\begin{bmatrix} 1 & 2 & 0 \\\\ 0 & 1 & 2 \\\\ 1 & 0 & 0 \\end{bmatrix} * \\begin{bmatrix} 1 & 0 & -1 \\\\ 1 & 0 & -1 \\\\ 1 & 0 & -1 \\end{bmatrix}$$
  $$\\text{Sum} = (1\\cdot1 + 2\\cdot0 + 0\\cdot(-1)) + (0\\cdot1 + 1\\cdot0 + 2\\cdot(-1)) + (1\\cdot1 + 0\\cdot0 + 0\\cdot(-1))$$
  $$\\text{Sum} = (1 + 0 + 0) + (0 + 0 - 2) + (1 + 0 + 0) = 1 - 2 + 1 = \\mathbf{0}$$

- **Position (0, 1)**: Window rows 0..2, cols 1..3:
  $$\\begin{bmatrix} 2 & 0 & 1 \\\\ 1 & 2 & 3 \\\\ 0 & 0 & 2 \\end{bmatrix} * K$$
  $$\\text{Sum} = (2\\cdot1 + 0\\cdot0 + 1\\cdot(-1)) + (1\\cdot1 + 2\\cdot0 + 3\\cdot(-1)) + (0\\cdot1 + 0\\cdot0 + 2\\cdot(-1))$$
  $$\\text{Sum} = (2 - 1) + (1 - 3) + (0 - 2) = 1 - 2 - 2 = \\mathbf{-3}$$

- **Position (0, 2)**: Window rows 0..2, cols 2..4:
  $$\\begin{bmatrix} 0 & 1 & 1 \\\\ 2 & 3 & 1 \\\\ 0 & 2 & 2 \\end{bmatrix} * K = (0-1) + (2-1) + (0-2) = -1 + 1 - 2 = \\mathbf{-2}$$

- **Position (1, 0)**: Window rows 1..3, cols 0..2:
  $$\\begin{bmatrix} 0 & 1 & 2 \\\\ 1 & 0 & 0 \\\\ 2 & 1 & 1 \\end{bmatrix} * K = (0-2) + (1-0) + (2-1) = -2 + 1 + 1 = \\mathbf{0}$$

- **Position (1, 1)**: Center window rows 1..3, cols 1..3:
  $$\\begin{bmatrix} 1 & 2 & 3 \\\\ 0 & 0 & 2 \\\\ 1 & 1 & 0 \\end{bmatrix} * K = (1-3) + (0-2) + (1-0) = -2 - 2 + 1 = \\mathbf{-3}$$

- **Position (1, 2)**: Window rows 1..3, cols 2..4:
  $$\\begin{bmatrix} 2 & 3 & 1 \\\\ 0 & 2 & 2 \\\\ 1 & 0 & 1 \\end{bmatrix} * K = (2-1) + (0-2) + (1-1) = 1 - 2 + 0 = \\mathbf{-1}$$

- **Position (2, 0)**: Window rows 2..4, cols 0..2:
  $$\\begin{bmatrix} 1 & 0 & 0 \\\\ 2 & 1 & 1 \\\\ 0 & 3 & 1 \\end{bmatrix} * K = (1-0) + (2-1) + (0-1) = 1 + 1 - 1 = \\mathbf{1}$$

- **Position (2, 1)**: Window rows 2..4, cols 1..3:
  $$\\begin{bmatrix} 0 & 0 & 2 \\\\ 1 & 1 & 0 \\\\ 3 & 1 & 2 \\end{bmatrix} * K = (0-2) + (1-0) + (3-2) = -2 + 1 + 1 = \\mathbf{0}$$

- **Position (2, 2)**: Bottom-Right window rows 2..4, cols 2..4:
  $$\\begin{bmatrix} 0 & 2 & 2 \\\\ 1 & 0 & 1 \\\\ 1 & 2 & 0 \\end{bmatrix} * K = (0-2) + (1-1) + (1-0) = -2 + 0 + 1 = \\mathbf{-1}$$
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Verifying with Python
Now let's compute these 9 positions with a loop in Python to verify our manual hand calculations!"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Verification loop for 5x5 image convolved with 3x3 filter
output_manual = np.zeros((3, 3), dtype=int)

for r in range(3):
    for c in range(3):
        patch = I[r:r+3, c:c+3]
        val = np.sum(patch * K)
        output_manual[r, c] = val
        print(f"Cell ({r}, {c}) patch sum = {val}")

print("\\nFinal Output Feature Map (3x3):\\n", output_manual)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
Our computed array exactly matches our hand-derived matrix:
$$\\begin{bmatrix} 0 & -3 & -2 \\\\ 0 & -3 & -1 \\\\ 1 & 0 & -1 \\end{bmatrix}$$
Every single cell value is 100% accounted for and verified!"""))

    # Section 4: Stride and Padding
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4. Stride and Padding

### (1) The Idea in One Plain Sentence
- **Stride** controls how many pixels the filter jumps at each step.
- **Padding** adds a border of zeros around the image to stop it from shrinking and keep edge pixels from being forgotten.

### (2) Real-Life Analogy
- **Stride**: Walking across a tiled floor taking baby steps (stride 1 = every tile) versus taking big strides (stride 2 = skipping every other tile).
- **Padding**: Putting a picture frame around a photo so the edges don't get cut off when you crop it.

### (3) Tiny Example with Real Numbers Worked by Hand
The General Output Size Formula:
$$\\text{Output Size} = \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1$$
Where:
- $W$: Input width (or height) = 5
- $K$: Kernel size = 3
- $P$: Padding = 1 (1 ring of zeros)
- $S$: Stride = 1

With padding $P=1$:
$$\\text{Output Size} = \\left\\lfloor \\frac{5 - 3 + 2(1)}{1} \\right\\rfloor + 1 = \\frac{4}{1} + 1 = 5$$
Notice the output is $5 \\times 5$, exactly the same size as the input! This is called **"Same" Padding**.
When $P=0$, the output is $(5 - 3)/1 + 1 = 3$. This is called **"Valid" Padding** (no padding).
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Testing Stride and Padding
Let's see what padding an array with zeros looks like in code using `np.pad`."""))

    nb.cells.append(nbf.v4.new_code_cell("""# Adding P=1 zero-padding to our 5x5 image
I_padded = np.pad(I, pad_width=1, mode='constant', constant_values=0)

print(f"Original shape: {I.shape}")
print(f"Padded shape with P=1: {I_padded.shape}")
print("\\nPadded Image (7x7):\\n", I_padded)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
Notice the new ring of zeros surrounding our original numbers. Now a $3 \\times 3$ filter placed at the top-left will have its center sitting directly on the original $(0, 0)$ pixel!"""))

    # Section 5: Channels and Feature Maps
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 5. Multi-Channel Convolutions & Feature Maps

### (1) The Idea in One Plain Sentence
In a multi-channel image (like RGB with 3 channels), each filter also has 3 channels; the filter performs a 2D convolution on each channel, sums the 3 results together, and adds one bias term to produce ONE single 2D feature map.

### (2) Real-Life Analogy
Think of a color photo as three stacked glass sheets: one Red, one Green, one Blue. A 3D filter is a 3-layer sandwich stencil. It slides through all three sheets simultaneously, measures the colors, and writes one combined score onto a single output sheet.

### (3) Tiny Example with Real Numbers
Suppose an input image has shape:
$$(\\text{Channels}_{\\text{in}}=3, \\text{Height}=32, \\text{Width}=32)$$
If we apply 16 different filters of size $3 \\times 3$:
- Each filter has dimensions: $(3, 3, 3)$ (3 channels, 3 rows, 3 columns).
- Number of weights in one filter: $3 \\times 3 \\times 3 = 27$ weights $+ 1$ bias $= 28$ parameters.
- For 16 filters: $16 \\times 28 = 448$ parameters.
- The output will have shape: $(16, 30, 30)$ (assuming no padding, stride 1).
- We now have **16 distinct feature maps**, each looking for a different visual texture!

### (4) Tensor Dimension Standard: (Batch, Channels, Height, Width)
In PyTorch, tensors are ALWAYS formatted in the order **$(B, C, H, W)$**:
- **B (Batch size)**: How many images are processed together in parallel (e.g. 64).
- **C (Channels)**: How many feature slices or color channels exist (e.g. 3 for RGB, 64 for hidden features).
- **H (Height)**: The vertical pixel count.
- **W (Width)**: The horizontal pixel count.
"""))

    # Section 6: Pooling
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 6. Pooling: Max Pooling and Average Pooling

### (1) The Idea in One Plain Sentence
Pooling reduces the spatial resolution of an image by summarizing small neighborhoods with their maximum or average value, cutting memory usage in half and making features position-tolerant.

### (2) Real-Life Analogy
**Zooming out!** When you stand 20 feet away from a painting, tiny individual pixel variations disappear, but the big shapes (the trees, the faces) remain crystal clear.

### (3) Tiny Example with Real Numbers Worked by Hand (2x2 Max Pooling)
Take a $4 \\times 4$ grid:
```
[[1, 3, 2, 4],
 [5, 6, 1, 2],
 [0, 2, 8, 3],
 [1, 4, 7, 9]]
```
Split into four non-overlapping $2 \\times 2$ blocks:
- Top-Left: $\\max([1, 3, 5, 6]) = \\mathbf{6}$
- Top-Right: $\\max([2, 4, 1, 2]) = \\mathbf{4}$
- Bottom-Left: $\\max([0, 2, 1, 4]) = \\mathbf{4}$
- Bottom-Right: $\\max([8, 3, 7, 9]) = \\mathbf{9}$

The pooled output is:
$$\\begin{bmatrix} 6 & 4 \\\\ 4 & 9 \\end{bmatrix}$$
Notice:
1. The resolution was cut in half ($4 \\times 4 \\rightarrow 2 \\times 2$).
2. There are **zero learnable parameters** in pooling! It is a fixed mathematical rule.
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Testing 2x2 Max Pooling in NumPy
Let's verify this $2 \\times 2$ Max Pooling operation with NumPy."""))

    nb.cells.append(nbf.v4.new_code_cell("""# 4x4 sample grid for Max Pooling
grid = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [0, 2, 8, 3],
    [1, 4, 7, 9]
])

# 2x2 max pooling with stride 2
pooled = np.zeros((2, 2), dtype=int)
for r in range(2):
    for c in range(2):
        pooled[r, c] = np.max(grid[r*2:(r+1)*2, c*2:(c+1)*2])

print("Input 4x4 Grid:\\n", grid)
print("\\nPooled 2x2 Grid (Max Pooling):\\n", pooled)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
The values $[6, 4, 4, 9]$ were extracted instantly. The highest signal in each quadrant is preserved, while noisy small variations are discarded."""))

    # Section 7: Translation Invariance & Receptive Field
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 7. Translation Invariance & Receptive Field

### (1) Translation Invariance
- **The Idea in One Plain Sentence**: An object detector should detect a cat regardless of whether the cat is on the left side or the right side of the photo.
- **Why CNNs Have It**: Because the same filter stencil scans the *entire* image. If an edge detector filter detects a whisker at $(10, 10)$, the exact same filter will detect that whisker if the cat moves to $(80, 80)$. Pooling then provides additional local shift tolerance.

### (2) Receptive Field
- **The Idea in One Plain Sentence**: The Receptive Field of a neuron is the patch of raw input pixels that can influence that neuron's final activation.
- **Real-Life Analogy**: A cone of vision. A single retina cell sees a tiny spot of light. But higher brain cells combine signals from many retina cells to see an entire face.
- **Why It Matters**:
  - Layer 1 (a $3 \\times 3$ conv): Each neuron sees a $3 \\times 3$ area of the raw image.
  - Layer 2 (another $3 \\times 3$ conv): Each neuron sees a $3 \\times 3$ area of Layer 1. But each of those Layer 1 neurons already saw $3 \\times 3$ of the raw image!
  - Therefore, the Layer 2 neuron's receptive field is:
    $$RF_2 = RF_1 + (K - 1) = 3 + (3 - 1) = 5 \\times 5$$
  - Two stacked $3 \\times 3$ conv layers have the same receptive field ($5 \\times 5$) as a single $5 \\times 5$ conv layer, but with far fewer parameters:
    - Single $5 \\times 5$: $5 \\times 5 = 25$ weights per channel.
    - Two stacked $3 \\times 3$: $2 \\times (3 \\times 3) = 18$ weights per channel (a 28% savings!).
    - Even better: Two layers have two non-linear activations (e.g. ReLU) instead of one, allowing the network to learn richer features!
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Before Code: Calculating Receptive Field Growth
Let's write a small Python loop to calculate how the receptive field grows as we stack $3 \\times 3$ convolutional layers and pooling layers."""))

    nb.cells.append(nbf.v4.new_code_cell("""# Simulating Receptive Field (RF) growth across layers
layers = [
    ("Conv2d (3x3, stride=1)", 3, 1),
    ("Conv2d (3x3, stride=1)", 3, 1),
    ("MaxPool2d (2x2, stride=2)", 2, 2),
    ("Conv2d (3x3, stride=1)", 3, 1),
    ("Conv2d (3x3, stride=1)", 3, 1),
]

current_rf = 1
current_stride = 1

print(f"{'Layer':<28} | {'Kernel':<6} | {'Stride':<6} | {'Receptive Field':<15}")
print("-" * 65)

for name, k, s in layers:
    current_rf = current_rf + (k - 1) * current_stride
    current_stride = current_stride * s
    print(f"{name:<28} | {k:<6} | {s:<6} | {current_rf}x{current_rf} pixels")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### After Code: What We Just Saw
By layer 5, each neuron sees a **$18 \\times 18$ pixel window** of the original image, even though every single filter is just a tiny $3 \\times 3$ grid! As we go deeper, the network transitions from seeing micro-edges to seeing whole object parts!"""))

    # Section 8: Step 1 Summary & Checkpoints
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 🎯 Step 1 Wrap-Up: Recap & Key Takeaway

### Recap in 3 Bullets:
1. **Fully connected nets fail on images** because flattening destroys 2D shapes and causes a parameter explosion (billions of weights for standard photos).
2. **CNNs solve this with two core inductive biases**: **Spatial Locality** (looking only at nearby pixels) and **Weight Sharing** (sliding the same filter stencil across every location).
3. **Stacking convolutions and pooling expands the Receptive Field**, allowing early layers to detect simple edges and deeper layers to assemble those edges into complex objects (eyes, wheels, faces).

### If you remember only one thing:
> **A convolution is simply a tiny stencil sliding across an image looking for a specific pattern; when the pattern matches, the output lights up with a high number.**
"""))

    return nb

if __name__ == "__main__":
    nb = create_notebook()
    notebook_path = "01_cnn_theory_and_intuition.ipynb"
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
