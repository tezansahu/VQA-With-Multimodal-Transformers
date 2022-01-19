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

## 🧪 Experimentation

### ⚡ Pipeline & Scripts
The `src/` folder contains all the scripts necessary for data processing & model training. All the configs & hyperparameters are specified in teh `params.yaml` file.

Following are the important scripts to experiment with the VQA models:

- `src/process_data.py`: Process the raw DAQUAR dataset available in `dataset/` folder & split it into training & evaluation sets, along with the space of all possible answers
- `src/main.py`: Train & evaluate the multimodal VQA model after loading the processed dataset.
- `src/inference.py`: Use a trained multimodal VQA model from a checkpoint to answer a question, given a reference image

After making necessary changes to the `params.yaml` file, the pipeline can be automated by running `dvc repro`. This will run the data processing & model training (& evaluation) stages.

For inferencing, run `python src/inference.py --config=params.yaml --img_path=<path-to-image> --question=<question>`

### 🤗 Models for Experimentation:

- Text Transformers (for encoding questions):
    - BERT (Bidirectional Encoder Representations from Transformers): `'bert-base-uncased'`
    - RoBERTa (Robustly Optimized BERT Pretraining Approach): `'roberta-base'`
    - ALBERT (A Lite BERT): `'albert-base-v2'`
- Image Transformers (for encoding images):
    - ViT (Vision Transformer): `'google/vit-base-patch16-224-in21k'`
    - DeiT (Data-Efficient Image Transformer): `'facebook/deit-base-distilled-patch16-224'`
    - BEiT (Bidirectional Encoder representation from Image Transformers): `'microsoft/beit-base-patch16-224-pt22k-ft22k'`


### 📊 VQA Performance of Various Models:

| Text Transformer | Image Transformer | Wu & Palmer Score | Accuracy | F1 | No. of Trainable Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | ViT | 0.286 | 0.235 | 0.020 | 197M |
| BERT | DeiT | 0.297 | 0.246 | 0.027 | 197M |
| BERT | BEiT | 0.303 | 0.254 | **0.034** | 196M |
| RoBERTa | ViT | 0.294 | 0.246 | 0.025 | 212M |
| RoBERTa | DeiT | 0.291 | 0.242 | 0.028 | 212M |
| RoBERTa | BEiT | _**0.308**_ | _**0.261**_ | 0.033 | 211M |
| ALBERT | ViT | 0.265 | 0.215 | 0.018 | 99M |
| ALBERT | DeiT | 0.140 | 0.085 | 0.002 | 99M |
| ALBERT | BEiT | 0.220 | 0.162 | 0.017 | 98M |

***

<p align="center">Created with ❤️ by <a href="https://www.linkedin.com/in/tezan-sahu/">Tezan Sahu</a></p>


