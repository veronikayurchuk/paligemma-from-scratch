Multimodal (Vision + Language) models from scratch in PyTorch.

In this repository, I will implement PaliGemma VLLM from scratch.
It is a lightweight multimodal model that consist of two parts: a vision encoder (Siglip) and a language encoder (Gemma).

Resources:
- [PaliGemma: A Lightweight Multimodal Model](https://arxiv.org/pdf/2407.07726)
- [Siglip: A Lightweight Vision Encoder](https://arxiv.org/pdf/2303.15343)
- [Gemma: A Lightweight Language Encoder](https://arxiv.org/pdf/2403.08295)


Overview:
![PaliGemma Architecture](images/paligemma_architecture.png)


There are 3 steps to implement PaliGemma. Each step will be accompanied by two files:
1. A notebook with detailed explanation and visualizations.
2. A python file with the implementation.

Here are the main steps:
Step 1: Implement Siglip. Follow modeling_siglip.py and modeling_siglip_explanation.ipynb.
Step 2: Implement Gemma. Follow modeling_gemma.py and modeling_gemma_explanation.ipynb.
Step 3: Implement PaliGemma. Follow modeling_pali_gemma.py and modeling_pali_gemma_explanation.ipynb.