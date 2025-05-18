# PaliGemma Implementation from Scratch

A PyTorch implementation of PaliGemma, a lightweight multimodal model that combines vision and language understanding capabilities. This project implements the model architecture from scratch, providing detailed explanations and visualizations for each component.

## 📝 Description

PaliGemma is a powerful yet lightweight multimodal model that consists of two main components:
- **Vision Encoder (SigLIP)**: A lightweight vision transformer for image understanding
- **Language Decoder (Gemma)**: A lightweight language model for text processing

This implementation serves as both an educational resource and a practical implementation guide for understanding multimodal models.

## 🎯 Project Goals

- Implement PaliGemma architecture from scratch in PyTorch
- Provide detailed explanations and visualizations for each component
- Create a modular and well-documented codebase
- [TODO] Add implementation of Gemma 2.
- [TODO] Add finetunning example with both code and detailed explanation

## 📚 Resources

### Research Papers
- [PaliGemma: A versatile 3B VLM for multimodal tasks](https://arxiv.org/pdf/2407.07726)
- [SigLIP: A Lightweight Vision Encoder](https://arxiv.org/pdf/2303.15343)
- [Gemma: Open Models Based on Gemini](https://arxiv.org/pdf/2403.08295)

## 🏗️ Architecture

![PaliGemma Architecture](/images/paligemma.png)

## 📋 Implementation Steps

The implementation is divided into three main steps, each with detailed explanations and code:

1. **Vision Encoder (SigLIP)**
   - File: `modeling_siglip.py`
   - Explanation: `modeling_siglip_explanation.ipynb`
   - Focus: Implementation of the vision encoder 

2. **Language Decoder (Gemma)**
   - File: `modeling_gemma.py`
   - Explanation: `modeling_gemma_explanation.ipynb`
   - Focus: Implementation of the language model architecture

3. **PaliGemma Integration**
   - TBD



---
*Note: This is a work in progress. More sections and details will be added as the implementation progresses.*