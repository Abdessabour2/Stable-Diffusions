# Project Title: Implementing and Applying Stable Diffusion Models

## 1. Introduction
This project explores the fascinating world of Stable Diffusion, a powerful class of generative models for image synthesis. The primary goal is to gain a deep understanding of Stable Diffusion's architecture and capabilities, achieved through two main approaches:
1.  Implementing core components of a Stable Diffusion model from scratch using PyTorch.
2.  Utilizing pre-built Hugging Face `diffusers` pipelines to develop a practical application for generating storyboards from textual narratives.

This work was undertaken as part of the "Advanced Learning INE2-DATA 2025 - Deep Learning Lab" project. It demonstrates both foundational knowledge of the model's internals and the ability to apply these advanced models to creative tasks.

## 2. Project Structure
The project is organized into two main Jupyter notebooks, each addressing a distinct aspect of Stable Diffusion:

* **Part 1: Stable Diffusion from Scratch (`stable_diffusion_scratch.ipynb`)**
    * **Objective**: To deconstruct and understand the fundamental building blocks of Stable Diffusion by implementing them individually. This notebook aims to provide a low-level perspective on how these models function.
    * **Key Components Implemented**:
        * Attention Mechanisms: Self-Attention and Cross-Attention.
        * Variational Autoencoder (VAE): Encoder, Decoder, and constituent blocks (Residual, Attention).
        * CLIP Text Encoder: Embedding layers and Transformer blocks (CLIPLayer).
        * UNet Denoising Model: Time Embedding, UNet Residual and Attention blocks, Upsampling/Downsampling, and the overall UNet structure.
        * DDPM Sampler: Logic for the diffusion and reverse diffusion (denoising) process.
        * A custom `generate` pipeline to tie these components together for image generation.

* **Part 2: Story-to-Storyboard with Hugging Face (`storyline.ipynb`)**
    * **Objective**: To leverage the high-level Hugging Face `diffusers` library to build a practical application capable of generating a sequence of images (a storyboard) from a given textual story.
    * **Workflow**:
        * **Story Input & Segmentation**: Takes a multi-sentence story as input and segments it into smaller, coherent scenes or narrative units (e.g., using `nltk.sent_tokenize` and custom chunking logic like the `segment_story` function).
        * **Visual Prompt Generation**: Each story segment is transformed into a descriptive visual prompt, potentially using text summarization (via a Hugging Face summarization pipeline) and appending artistic style keywords (as seen in `generate_visual_prompts`).
        * **Image Generation**: Utilizes a pre-trained Stable Diffusion model (e.g., `runwayml/stable-diffusion-v1-5`) via the `StableDiffusionPipeline` from `diffusers` to generate an image for each visual prompt.
        * **Storyboard Display**: The generated images are then displayed in sequence, often alongside their corresponding prompts or scene descriptions, forming a visual storyboard.

## 3. Environment Setup
* **Python version**: Python 3.9+ is recommended.
* **Dependencies**:
    * All dependencies are listed in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
* **Model Checkpoints & Tokenizers**:
    * **For `storyline .ipynb`**:
        * The Hugging Face `diffusers` library will automatically download the required pre-trained model (e.g., `runwayml/stable-diffusion-v1-5` or `stabilityai/stable-diffusion-2-1-base`, as specified in the notebook) and its associated tokenizer on the first run. An active internet connection is required for this initial download.

## 4. Source Code Documentation

This project implements and utilizes several key deep learning components. Below is a brief overview:

* **Attention Mechanisms (`SelfAttention`, `CrossAttention`)**:
    * These are fundamental building blocks in Transformer architectures, including CLIP and the UNet in Stable Diffusion. Self-attention allows tokens within a sequence to weigh the importance of other tokens in the same sequence. Cross-attention enables one sequence to draw information from another (e.g., image features attending to text prompt embeddings). The implementation follows the scaled dot-product attention mechanism.

* **Variational Autoencoder (VAE)**:
    * The VAE (`VAE_Encoder`, `VAE_Decoder`) is used to compress images from pixel space into a lower-dimensional latent space and to decompress latents back into images. The diffusion process operates in this more manageable latent space, making training and inference more efficient. The VAE architecture includes residual blocks and attention blocks.

* **CLIP Text Encoder (`CLIPEmbedding`, `CLIPLayer`, `CLIP`)**:
    * The Contrastive Language-Image Pre-Training (CLIP) model's text encoder is used to convert textual prompts into rich semantic embeddings. These embeddings serve as the conditioning signal for the UNet, guiding the image generation process. The implementation consists of token/positional embeddings followed by a stack of Transformer encoder layers.

* **UNet Denoising Model (`TimeEmbedding`, `UNET_ResidualBlock`, `UNET_AttentionBlock`, `UNET`)**:
    * The UNet is the core of the diffusion model. It's an encoder-decoder architecture with skip connections, designed to predict the noise present in a noisy latent representation at a given timestep. It takes the noisy latents, the timestep embedding, and the CLIP text embeddings (via cross-attention) as input.

* **DDPMSampler**:
    * This class implements the Denoising Diffusion Probabilistic Models (DDPM) sampling logic. It defines the noise schedule (betas, alphas, alphas_cumprod) and provides the `step` method to perform one reverse diffusion (denoising) step. It also includes an `add_noise` method for the forward diffusion process.

* **Main Generation Pipelines**:
    * `stable_diffusion_scratch (2).ipynb` contains a `generate` function that orchestrates the from-scratch components: encoding prompts with CLIP, preparing initial latents (random or from an image), iteratively denoising using the UNet and DDPMSampler, and finally decoding with the VAE.
    * `storyline (1).ipynb` uses helper functions (`segment_story`, `generate_visual_prompts`, `generate_image`, `display_storyboard`) to process a story, generate prompts, call the Hugging Face `StableDiffusionPipeline` for each prompt, and display the resulting images as a storyboard.

Detailed explanations and comments are provided within the Jupyter notebooks for each implemented class and function.

## 5. Example Outputs

![{A72D2130-CDAA-49CC-9EC9-F7CF9F36C684}](https://github.com/user-attachments/assets/5794f683-cd41-4209-89ec-4de9492b9cca)

