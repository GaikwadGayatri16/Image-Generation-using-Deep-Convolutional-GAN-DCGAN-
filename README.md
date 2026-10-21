# Generating Kuzushiji Characters Using a Deep Convolutional GAN (DCGAN)

## Practical No. 04 — Generative AI

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented using TensorFlow and Keras to generate synthetic images of classical Japanese cursive characters from the Kuzushiji-MNIST (KMNIST) dataset.

---

## Student Information

| Field | Details |
|---|---|
| **Name** | Gayatri Pandharinath Gaikwad |
| **Roll No. / PRN** | 202502110002 |
| **Division / Batch** | A3 |
| **Subject** | Generative AI |
| **Practical No.** | 04 |
| **Dataset** | Kuzushiji-MNIST (KMNIST) |
| **Model** | Deep Convolutional Generative Adversarial Network (DCGAN) |
| **Framework** | TensorFlow / Keras |

---

## 1. Project Overview

Generative Artificial Intelligence focuses on developing models that can learn patterns from existing data and generate new samples with similar characteristics.

In this practical, a **Deep Convolutional Generative Adversarial Network (DCGAN)** is implemented and trained on the Kuzushiji-MNIST dataset. The dataset contains grayscale images of Japanese Hiragana characters written in a classical cursive style.

The model learns the visual patterns and structures of these characters and generates new character-like images from random noise.

The project also studies the training process through loss curves, generated-image visualizations, and quantitative evaluation metrics.

## 2. Objectives

- Understand the working principles of Generative Adversarial Networks (GANs).
- Implement a DCGAN using TensorFlow and Keras.
- Explore and preprocess the KMNIST dataset.
- Build and train a Generator and a Discriminator.
- Generate synthetic Japanese character images from random noise.
- Analyze Generator and Discriminator losses during training.
- Evaluate generated images for quality, diversity, and class coverage.
- Understand limitations such as blurry images and mode collapse.

## 3. Dataset Description

### Kuzushiji-MNIST (KMNIST)

KMNIST is an image dataset designed for machine learning research involving classical Japanese cursive characters. It provides an alternative to the traditional MNIST handwritten-digit dataset.

**Dataset characteristics:**

- Total images: 70,000
- Training images: 60,000
- Test images: 10,000
- Image dimensions: 28 × 28 pixels
- Image format: Grayscale
- Number of classes: 10
- Training images per class: 6,000

The ten classes are represented by the following romanized labels:

`o`, `ki`, `su`, `tsu`, `na`, `ha`, `ma`, `ya`, `re`, `wo`

### Dataset preprocessing

The image pixels are converted to `float32` and normalized from the range `[0, 255]` to `[-1, 1]` using:

```python
x_scaled = (x - 127.5) / 127.5
```

This normalization matches the output range of the Generator's `tanh` activation function.

The notebook loads KMNIST using TensorFlow Datasets, with a fallback to the dataset's direct download files if the initial loading method fails.

## 4. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning and model training |
| Keras | Neural network construction |
| NumPy | Numerical operations and random noise generation |
| Pandas | Organizing training metrics and evaluation results |
| Matplotlib | Visualizing images, loss curves, and evaluation results |
| SciPy | Matrix operations used in feature-based evaluation |
| Pillow (PIL) | Image processing and GIF creation |
| TensorFlow Datasets | Loading the KMNIST dataset |
| Jupyter Notebook / Google Colab | Interactive execution and experimentation |
| GitHub | Version control and project hosting |

## 5. Important Concepts

### 5.1 Generative Adversarial Network (GAN)

A GAN consists of two neural networks that compete with each other.

**Generator (G):** Takes a random noise vector as input and produces a synthetic image.

**Discriminator (D):** Receives real or generated images and predicts whether an image is real or fake.

During training, the Generator tries to fool the Discriminator, while the Discriminator learns to distinguish generated images from real training images.

This adversarial process encourages the Generator to learn the underlying visual patterns of the dataset.

### 5.2 Deep Convolutional GAN (DCGAN)

DCGAN applies convolutional neural network techniques to GANs, making them suitable for image generation.

The Generator uses dense and transposed-convolution layers to construct an image from random noise. The Discriminator uses convolutional layers to extract image features and classify an image as real or generated.

### 5.3 Random Noise and Latent Space

The Generator receives a 100-dimensional random noise vector sampled from a normal distribution.

This vector is called a latent vector. Different vectors can produce different images, allowing the model to generate new samples.

The notebook also explores latent-space interpolation by gradually moving between two noise vectors and observing how the generated images change.

### 5.4 Loss Functions

Binary cross-entropy is used to calculate the training losses.

- **Discriminator loss:** Measures how well the Discriminator distinguishes real images from generated images.
- **Generator loss:** Measures how effectively the Generator produces images that the Discriminator classifies as real.

The notebook uses one-sided label smoothing, setting the real target to `0.9` rather than `1.0`, to help prevent the Discriminator from becoming excessively confident.

## 6. Model Architecture

### Generator

The Generator transforms a 100-dimensional noise vector into a 28 × 28 grayscale image.

Its main architecture is:

1. Input: 100-dimensional random noise vector.
2. Dense layer: Projects the noise into a feature representation.
3. Reshape: Converts the features into a 7 × 7 × 128 tensor.
4. Transposed convolution: Upsamples the features to 14 × 14 × 128.
5. Transposed convolution: Upsamples the features to 28 × 28 × 64.
6. Final convolution: Produces a single-channel grayscale image.
7. `tanh` activation: Restricts output pixels to the range `[-1, 1]`.

Batch normalization and LeakyReLU activations are used in the Generator to support stable training.

### Discriminator

The Discriminator accepts a 28 × 28 × 1 image and predicts the probability that it is real.

Its main architecture is:

1. Input: 28 × 28 grayscale image.
2. Convolutional layer: 64 filters.
3. Convolutional layer: 128 filters with stride 2.
4. Convolutional layer: 128 filters with stride 2.
5. Flatten layer: Converts feature maps into a vector.
6. Dropout: Uses a dropout rate of `0.4`.
7. Dense output layer: Produces a probability using sigmoid activation.

LeakyReLU is used after the convolutional layers.

## 7. Training Configuration

The notebook uses the following hyperparameters:

| Parameter | Value |
|---|---:|
| Latent dimension | 100 |
| Number of epochs | 50 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam beta 1 | 0.5 |
| Real-label smoothing target | 0.9 |
| Discriminator dropout | 0.4 |
| LeakyReLU negative slope | 0.2 |
| Generator checkpoint interval | Every 10 epochs |
| Random seed | 2026 |

Both networks are updated during each training step using separate gradient calculations and separate Adam optimizers. This ensures that each network is updated according to its own loss function.

The notebook records training metrics for each iteration and calculates epoch-level averages.

## 8. Project Workflow

The practical follows these stages:

1. **Environment setup:** Import libraries, set random seeds, configure hyperparameters, and prepare the output directory.
2. **Dataset exploration:** Load KMNIST and inspect sample images, class distribution, and pixel statistics.
3. **Preprocessing:** Normalize image values to the range `[-1, 1]`.
4. **Model construction:** Build the Generator and Discriminator.
5. **GAN training:** Train both networks adversarially for 50 epochs.
6. **Progress monitoring:** Record loss values, accuracy diagnostics, and gradient norms.
7. **Image generation:** Generate new character images from random noise.
8. **Visualization:** Compare real and generated images and inspect changes across training epochs.
9. **Quantitative evaluation:** Examine classifier-based measures, class coverage, feature distributions, and image diversity.
10. **Result export:** Save model checkpoints, training history, visualizations, and a ZIP archive of generated outputs.

## 9. Image Generation and Visualization

The trained Generator is used for several experiments:

- Generate a grid of 100 synthetic character images using fresh random noise.
- Compare real test images with generated samples.
- Visualize training progress using the same fixed noise vectors at selected epochs.
- Create an animation showing how generated samples evolve during training.
- Explore latent-space interpolation between random vectors.
- Study how changing the noise scale affects generated images.

These experiments help illustrate how the Generator learns character-like structures over time.

## 10. Evaluation Metrics

The notebook includes five main categories of quantitative evaluation.

**1. Discriminator evaluation**

Checks how the trained Discriminator classifies unseen real test images and newly generated images.

**2. Classifier confidence and Inception-style score**

A separate CNN trained on real KMNIST images is used to assess generated-image predictions and the balance between confidence and diversity.

**3. Class distribution and coverage**

Examines whether generated samples cover the ten character classes rather than concentrating on only a few classes.

**4. Fréchet-distance-style feature comparison**

Compares feature distributions from real and generated images using a custom Fréchet-distance calculation on the reference classifier's hidden features.

This is a KMNIST-specific, FID-style measure and should not be directly compared with standard published ImageNet-based FID scores.

**5. Diversity and memorization checks**

Compares pairwise image distances and nearest-neighbor relationships to investigate whether the Generator produces varied images or overly repetitive samples.

The evaluation results depend on the actual training run and should be interpreted alongside the generated-image grids and loss curves.

## 11. Outputs and Results

The notebook is designed to save several artifacts in the `kmnist_dcgan_outputs` directory, including:

- `generator_final.keras` — final Generator model.
- `discriminator_final.keras` — final Discriminator model.
- `generator_epoch_010.weights.h5`, `generator_epoch_020.weights.h5`, and subsequent checkpoints — saved Generator weights at configured intervals.
- `epoch_history.csv` — epoch-level training metrics.
- `final_generated_100.png` — grid of 100 generated character images.
- `real_vs_generated.png` — comparison of real and generated images.
- `training_progress.png` — visualization of generated samples across training.
- Additional evaluation plots and a ZIP archive containing the output artifacts.

The notebook also creates a GIF of generated-image progress and displays a summary of the training and evaluation metrics.

Actual accuracy, loss, and image-quality values should be taken from the results produced when the notebook is executed.

## 12. Limitations

- The model is an **unconditional GAN**; it cannot directly generate a requested character class.
- Some generated strokes may be blurry, incomplete, or disconnected.
- Mode collapse may occur if the Generator produces limited varieties of images.
- Training is limited to 50 epochs in the configured experiment.
- Classifier-based evaluation metrics depend on the custom reference classifier and are not universal quality measures.
- GAN losses alone do not guarantee realistic or diverse generated images.

## 13. Future Improvements

Potential extensions include:

- Train for additional epochs and compare image quality.
- Experiment with different learning rates and batch sizes.
- Implement a Conditional GAN to generate specific character classes.
- Explore Wasserstein GAN with Gradient Penalty (WGAN-GP).
- Apply spectral normalization to improve Discriminator stability.
- Compare DCGAN outputs with other image-generation architectures.
- Investigate improved evaluation methods for image quality and diversity.

## 14. How to Run the Project

### Option A: Google Colab

1. Open Google Colab.
2. Upload `GAN_KMNIST_v2 (1).ipynb` or open it from your GitHub repository.
3. Select a suitable runtime; a GPU is recommended to reduce training time.
4. Run the notebook cells in order, starting from environment setup.
5. Allow the dataset to download and the models to train.
6. Inspect the loss curves, generated images, and evaluation summary.
7. Download the generated output ZIP if prompted.

### Option B: Local Python Environment

1. Install Python and create a virtual environment.
2. Install the required libraries.
3. Launch Jupyter Notebook or JupyterLab.
4. Open the notebook and run its cells in sequence.

Example installation command:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scipy pillow jupyter
```

**Note:** The notebook contains a setup cell for installing `importlib_resources` in compatible environments. GPU availability, TensorFlow compatibility, and dataset download access can affect execution time and setup.

## 15. Repository Structure

A suggested GitHub repository structure is:

```text
Image-Generation-using-Deep-Convolutional-GAN-DCGAN/
│
├── GAN_KMNIST_v2 (1).ipynb
├── README.md
└── kmnist_dcgan_outputs/
    ├── generator_final.keras
    ├── discriminator_final.keras
    ├── epoch_history.csv
    ├── final_generated_100.png
    ├── real_vs_generated.png
    └── training_progress.png
```

The output directory and its files are generated when the notebook runs. They do not need to be committed unless you choose to include the results in your repository.

## 16. Conclusion

This practical implements a Deep Convolutional Generative Adversarial Network using TensorFlow and Keras to generate synthetic images resembling classical Japanese cursive characters from KMNIST.

The experiment demonstrates the interaction between the Generator and Discriminator, the role of convolutional architectures in image generation, and the importance of monitoring losses, visual outputs, diversity, and class coverage.

The notebook also illustrates that GAN evaluation requires multiple forms of evidence: a loss curve by itself cannot establish whether generated images are realistic or sufficiently diverse.

## 17. References

1. Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems (NeurIPS).
2. Radford, A., Metz, L., and Chintala, S. (2016). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
3. Clanuwat, T. et al. (2018). *Deep Learning for Classical Japanese Literature: A Dataset*.
4. Salimans, T. et al. (2016). *Improved Techniques for Training GANs*. NeurIPS.
5. Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS.
6. TensorFlow and Keras documentation.

---

**Project:** Generating Kuzushiji Characters Using DCGAN  
**Practical:** No. 04  
**Subject:** Generative AI
