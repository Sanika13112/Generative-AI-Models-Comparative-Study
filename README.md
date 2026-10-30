# Comparative Study and Implementation of Generative AI Models

A practical comparative study of four Generative AI models — **Autoencoder (AE), Variational Autoencoder (VAE), Deep Convolutional Generative Adversarial Network (DCGAN), and a pretrained Diffusion Model** — using face images from the Labeled Faces in the Wild (LFW) dataset.

The project explores how different generative models reconstruct existing images and generate new images. It compares their architectures, working principles, output quality, loss behaviour, advantages, limitations, and computational requirements.

---

## Project Overview

Generative Artificial Intelligence (Generative AI) focuses on learning patterns and distributions from existing data to reconstruct inputs or generate new samples.

In this project, we implement and analyse four generative models:

| Model | Category | Main Purpose |
|---|---|---|
| Autoencoder (AE) | Encoder–Decoder Model | Image reconstruction |
| Variational Autoencoder (VAE) | Probabilistic Generative Model | Image reconstruction and latent-space generation |
| DCGAN | Adversarial Generative Model | Generation of new face images |
| Diffusion Model | Denoising Generative Model | Prompt-based image generation |

The project is divided into two main activities:

- **CO3 – Image Reconstruction:** Autoencoder and Variational Autoencoder.
- **CO4 – Image Generation:** DCGAN and pretrained Diffusion Model.

## Objectives

- Understand the architecture and working principles of four generative AI models.
- Implement Autoencoder and VAE for image reconstruction.
- Generate new images using DCGAN.
- Use a pretrained Diffusion Model for text-to-image generation.
- Compare reconstruction and generation quality.
- Analyse training losses, model behaviour and execution time.
- Understand the advantages, limitations and applications of different generative models.

---

## Dataset

**Dataset Name:** Labeled Faces in the Wild (LFW)

The LFW dataset contains face photographs of people collected for research in unconstrained face recognition.

Dataset source:

https://vis-www.cs.umass.edu/lfw/

The notebook downloads the dataset directly, extracts the images, selects a subset and preprocesses them.

### Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset | LFW |
| Selected Images | 3,000 |
| Image Dimensions | 64 × 64 |
| Colour Channels | RGB (3) |
| Pixel Normalization | 0 to 1 |
| Training Images | 2,700 |
| Validation Images | 300 |
| Total Images | 3,000 |

### Preprocessing Steps

1. Download the LFW dataset.
2. Extract the downloaded archive.
3. Collect image file paths.
4. Shuffle the image paths.
5. Select 3,000 images.
6. Convert images into RGB format.
7. Resize images to 64 × 64 pixels.
8. Normalize pixel values to the range 0–1.
9. Split the dataset into training and validation sets.

---

## Models Implemented

### 1. Autoencoder (AE)

An Autoencoder is a neural network that learns to compress an input image into a smaller latent representation and reconstruct the original image.

**Main Components:**
- Encoder
- Latent representation
- Decoder

**Working Principle:**

Input Image → Encoder → Latent Vector → Decoder → Reconstructed Image

**Implementation Configuration:**

| Parameter | Value |
|---|---|
| Latent Dimension | 128 |
| Epochs | 10 |
| Loss Function | Mean Squared Error (MSE) |
| Training Time | 31.1 seconds |

**Experimental Results:**

- Final Training Loss: 0.0130
- Initial Validation Loss: 0.0548
- Final Validation Loss: 0.0144

**Observation:**

The Autoencoder reconstructed the general face structure, skin tone and lighting. However, some fine details, such as hair texture and facial features, appeared blurred.

### 2. Variational Autoencoder (VAE)

A Variational Autoencoder is a probabilistic generative model that learns a structured latent space by representing input data using a probability distribution.

Unlike a traditional Autoencoder, the encoder produces a mean and log variance rather than a single latent vector.

**Main Components:**
- Encoder
- Mean and log variance
- Sampling layer
- Decoder

**Working Principle:**

Input Image → Encoder → Mean and Variance → Sampling → Decoder → Reconstructed Image

The VAE uses the reparameterization trick:

`z = μ + σ × ε`

where:
- μ represents the mean.
- σ represents the standard deviation.
- ε is sampled from a standard normal distribution.

**Implementation Configuration:**

| Parameter | Value |
|---|---|
| Latent Dimension | 128 |
| Epochs | 10 |
| Loss Function | Reconstruction Loss + KL Divergence |
| Training Time | 20.5 seconds |

**Experimental Results:**

- Total Loss: 6772.4150
- Reconstruction Loss: 6681.3657
- KL Divergence Loss: 91.0485

**Observation:**

The VAE generated smoother reconstructions than the Autoencoder. Its organised latent space supports sampling new images, although reconstruction details were softer.

### 3. Deep Convolutional Generative Adversarial Network (DCGAN)

DCGAN is a generative adversarial network that uses convolutional neural networks to generate realistic-looking images.

It consists of two neural networks that compete with each other.

**Main Components:**
- Generator
- Discriminator

**Working Principle:**

Random Noise → Generator → Fake Image

Real Image / Fake Image → Discriminator → Real or Fake Prediction

The generator tries to create realistic images, while the discriminator tries to distinguish real images from generated images.

**Implementation Configuration:**

| Parameter | Value |
|---|---|
| Noise Dimension | 100 |
| Epochs | 30 |
| Training Time | 333.6 seconds |
| Loss Function | Adversarial Binary Cross-Entropy |

**Experimental Results:**

- Generator Loss: 0.7858
- Discriminator Loss: 1.3831

**Observation:**

The DCGAN generated new face-like images from random noise. Some outputs contained colour noise, distortions and visible artefacts. Loss values alone cannot determine image quality, so the generated samples were also visually inspected.

### 4. Pretrained Diffusion Model

Diffusion models generate images by gradually removing noise from a randomly initialized representation.

In this project, we used the pretrained `segmind/tiny-sd` model. It was not trained on the LFW dataset.

**Main Components:**
- Text Encoder
- U-Net
- Noise Scheduler
- VAE Decoder

**Working Principle:**

Text Prompt + Random Noise → Iterative Denoising → VAE Decoder → Generated Image

**Model Configuration:**

| Parameter | Value |
|---|---|
| Pretrained Model | segmind/tiny-sd |
| Platform | Hugging Face Diffusers |
| Inference Steps | 20 |
| Guidance Scale | 7.0 |
| Device | CUDA |
| Generation Time | 3.34 seconds |

**Prompt Used:**

"A realistic portrait photograph of a person, soft natural daylight, neutral background, high detail."

**Observation:**

The pretrained Diffusion Model generated a detailed portrait with realistic visual features. Its output quality was higher in this experiment, but the comparison is not like-for-like because the model uses pretrained weights from a much larger training process.

---

## Experimental Results and Comparison

### Performance Summary

| Model | Epochs / Steps | Execution Time | Final Result |
|---|---:|---:|---|
| Autoencoder | 10 epochs | 31.1 sec | Validation MSE: 0.0144 |
| VAE | 10 epochs | 20.5 sec | Total Loss: 6772.4150 |
| DCGAN | 30 epochs | 333.6 sec | G: 0.7858, D: 1.3831 |
| Diffusion | 20 inference steps | 3.34 sec | Prompt-generated portrait |

**Important:** AE, VAE and DCGAN execution times represent training time. The Diffusion Model's 3.34 seconds represents inference time using pretrained weights. These times should not be interpreted as directly comparable training costs.

### Key Observations

- AE reconstructed the overall input image but lost some fine details.
- VAE produced smoother reconstructions and learned an organised latent space.
- DCGAN generated new face-like images but showed visible artefacts.
- Pretrained Diffusion generated more detailed and realistic images in this experiment.
- Different loss functions have different scales, so their numerical values should not be directly compared.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Notebook execution environment |
| TensorFlow / Keras | AE, VAE and DCGAN implementation |
| PyTorch | Pretrained Diffusion model execution |
| NumPy | Numerical operations |
| Matplotlib | Graphs and visualisation |
| PIL (Pillow) | Image loading and preprocessing |
| Hugging Face Diffusers | Pretrained Diffusion pipeline |
| LFW Dataset | Face image dataset |

---

## Project Structure

A recommended GitHub repository structure is:

```text
Generative-AI-Assignment-02/
│
├── README.md
│
├── notebook/
│   └── Generative_AI_final_IA.ipynb
│
├── report/
│   └── GENAI_FINAL_REPORT.docx
│
├── presentation/
│   └── GenAI_PPT.pptx
│
├── results/
│   ├── dataset_samples.png
│   ├── autoencoder_reconstruction.png
│   ├── vae_reconstruction.png
│   ├── dcgan_generated_images.png
│   ├── dcgan_loss_graph.png
│   └── diffusion_generated_image.png
│
└── LICENSE
```

The folder and filenames above are suggested. Add only the files and screenshots you actually include in your repository.

---

## How to Run the Project

The complete implementation is provided in a Google Colab notebook.

### Option 1: Run using Google Colab

1. Open the notebook using the link below.
2. Sign in to your Google account.
3. Select **Runtime → Change runtime type**.
4. Select GPU if available.
5. Run the notebook cells in sequence.
6. Allow the notebook to download and preprocess the LFW dataset.
7. Execute the AE, VAE and DCGAN experiments.
8. Run the pretrained Diffusion pipeline.
9. Observe the reconstructed images, generated images and experimental results.

**Google Colab Notebook:**

[https://colab.research.google.com/drive/1t1PBKGPbLAadVUDPFQuYTkzCqsJB8EHi?usp=sharing](https://colab.research.google.com/drive/1t1PBKGPbLAadVUDPFQuYTkzCqsJB8EHi?usp=sharing)

### Option 2: Run Locally

Clone the repository:

```bash
git clone (https://github.com/Sanika13112/Generative-AI-Models-Comparative-Study)
```

Navigate to the project folder:

```bash
cd Generative-AI-Assignment-02
```

Install the required Python packages:

```bash
pip install numpy matplotlib pillow tensorflow torch torchvision
pip install diffusers transformers accelerate
```

Install Jupyter Notebook if required:

```bash
pip install notebook
```

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebook and execute its cells.

**Note:** The notebook was developed in Google Colab. Local execution may require compatible versions of TensorFlow, PyTorch, CUDA and Hugging Face Diffusers. A GPU is recommended for faster execution of the Diffusion Model.

---

## Limitations

- Only 3,000 images from the LFW dataset were used.
- Images were resized to 64 × 64 for the AE, VAE and DCGAN experiments.
- AE and VAE were trained for only 10 epochs.
- DCGAN was trained for 30 epochs.
- The Diffusion Model was pretrained and was not trained on LFW.
- Image quality was evaluated visually; metrics such as FID and SSIM were not calculated.
- Each experiment was run once, so execution times may vary across hardware and sessions.

## Learning Outcomes

Through this assignment, we learned:

1. How encoder–decoder architectures perform image reconstruction.
2. How VAE uses probability distributions and the reparameterization trick.
3. How Generator and Discriminator networks work together in DCGAN.
4. How Diffusion Models generate images through iterative denoising.
5. Why reconstruction quality and image generation quality need different evaluation approaches.
6. Why training time and inference time must be compared separately.

---

## Group Members

| Name | PRN |
|---|---|
| Amruta Kanase | 202401110042 |
| Sanika Mane | 202401110047 |
| Tanishka Tapkir | 202401110071 |

**Institution:** MIT Academy of Engineering, Alandi  
**Department:** Artificial Intelligence and Machine Learning  
**Assignment:** Internal Assignment – 02  
**Subject:** Generative AI

### Contributions

- **Sanika and Tanishka:** AE and VAE experiments, reconstruction screenshots and CO3 analysis.
- **Tanishka and Amruta:** DCGAN and pretrained Diffusion experiments, generated outputs and CO4 analysis.
- **Amruta and Sanika:** Comparative analysis, AI documentation, verification, report and PPT preparation.

---

## AI Tools and Verification

**AI Tool Used:** Claude

AI assistance was used for understanding model concepts, explaining notebook code, preparing the comparative analysis and structuring the report.

The group verified experimental values against the executed Google Colab notebook and cross-checked theoretical explanations using original research papers and official documentation.

---

## References

1. Huang, G. B. et al. (2007). *Labeled Faces in the Wild*. University of Massachusetts Amherst.
   https://vis-www.cs.umass.edu/lfw/

2. Kingma, D. P. and Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR.

3. Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS.

4. Radford, A., Metz, L. and Chintala, S. (2016). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR.

5. Ho, J., Jain, A. and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.

6. Segmind. *segmind/tiny-sd – Pretrained Diffusion Model*.
   https://huggingface.co/segmind/tiny-sd

7. TensorFlow Documentation:
   https://www.tensorflow.org/

8. Hugging Face Diffusers Documentation:
   https://huggingface.co/docs/diffusers/

---

## Acknowledgement

We sincerely thank our faculty and the Department of Artificial Intelligence and Machine Learning at MIT Academy of Engineering for providing the opportunity to study and experiment with Generative AI models.

This assignment helped us understand image reconstruction, image generation and the practical differences between major generative modelling approaches.

---

**If you find this project useful, feel free to explore the notebook and learn from the implementation.**
