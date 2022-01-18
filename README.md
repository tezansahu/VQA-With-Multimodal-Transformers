# Visual Question Answering with Multimodal Transformers

This repo contains the dataset & code for exploring multimodal *fusion-type* transformer models (from Huggingface 🤗) for the task of visual question answering.

## 🗂️ Dataset Used: [DAQUAR Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge/)

## ☑️ Requirements

Create a virtual environment & install the required packages using `pip install -r requirements.txt`:

In particular, we will require the following packages:

- `datasets==1.17.0`
- `nltk==3.5`
- `pandas==1.3.5`
- `Pillow==9.0.0`
- `scikit-learn==0.23.2`
- `torch==1.8.2+cu111`
- `transformers==4.15.0`
- `dvc==2.9.3` *(for automating the training pipeline)*

> _**Note:** It is best to have some GPU available to train the multimodal models (Google Colab can be used)._


## 📝 Notebook: [`VisualQuestionAnsweringWithTransformers.ipynb`](./notebooks/VisualQuestionAnsweringWithTransformers.ipynb)


## 🤗 Models for Experimentation:

- Text Transformers (for encoding questions):
    - BERT (Bidirectional Encoder Representations from Transformers): `'bert-base-uncased'`
    - RoBERTa (Robustly Optimized BERT Pretraining Approach): `'roberta-base'`
    - ALBERT (A Lite BERT): `'albert-base-v2'`
- Image Transformers (for encoding images):
    - ViT (Vision Transformer): `'google/vit-base-patch16-224-in21k'`
    - DeiT (Data-Efficient Image Transformer): `'facebook/deit-base-distilled-patch16-224'`
    - BEiT (Bidirectional Encoder representation from Image Transformers): `'microsoft/beit-base-patch16-224-pt22k-ft22k'`


## 📊 VQA Performance of Various Models:

| Text Transformer | Image Transformer | Wu & Palmer Score | Accuracy | F1 | No. of Trainable Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | ViT | 0.281 | 0.231 | 0.022 | 197M |
| BERT | DeiT | 0.295 | 0.245 | 0.027 | 197M |
| BERT | BEiT | 0.295 | 0.244 | 0.026 | 196M |
| RoBERTa | ViT | 0.287 | 0.244 | 0.026 | 212M |
| RoBERTa | DeiT | 0.296 | 0.246 | 0.029 | 212M |
| _**RoBERTa**_ | _**BEiT**_ | _**0.308**_ | _**0.259**_ | _**0.034**_ | _**211M**_ |
| ALBERT | ViT | 0.280 | 0.230 | 0.021 | 99M |
| ALBERT | DeiT | 0.200 | 0.151 | 0.014 | 99M |
| ALBERT | BEiT | 0.253 | 0.197 | 0.023 | 98M |

***

<p align="center">Created with ❤️ by <a href="https://www.linkedin.com/in/tezan-sahu/">Tezan Sahu</a></p>


