import argparse
import os
import yaml
import logging
from typing import Text, Dict, List
from PIL import Image

import torch
import transformers
from model import MultimodalVQAModel

def loadAnswerSpace(config: Dict) -> List[str]:
    with open(os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"])) as f:
        answer_space = f.read().splitlines()
    return answer_space

def tokenizeQuestion(config: Dict, question: Text, device) -> Dict:
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    encoded_text = tokenizer(
        text=[question],
        padding=config["tokenizer"]["padding"],
        max_length=config["tokenizer"]["max_length"],
        truncation=config["tokenizer"]["truncation"],
        return_tensors='pt',
        return_token_type_ids=config["tokenizer"]["return_token_type_ids"],
        return_attention_mask=config["tokenizer"]["return_attention_mask"],
    )
    return {
        "input_ids": encoded_text['input_ids'].to(device),
        "token_type_ids": encoded_text['token_type_ids'].to(device),
        "attention_mask": encoded_text['attention_mask'].to(device),
    }

def featurizeImage(config: Dict, img_path: Text, device) -> Dict:
    featurizer = transformers.AutoFeatureExtractor.from_pretrained(config["model"]["image_encoder"])
    processed_images = featurizer(
            images=[Image.open(img_path).convert('RGB')],
            return_tensors="pt",
        )
    return {
        "pixel_values": processed_images['pixel_values'].to(device),
    }


def main(config_path: Text, img_path: Text, question: Text) -> None:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    if config["base"]["use_cuda"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # SET ONLY 1 GPU DEVICE
    else:
        device =  torch.device('cpu')
    
    # Load the space of all potential answer 
    logging.info("Loading the space of all answers...")
    answer_space = loadAnswerSpace(config)
    
    # Tokenize the question & featurize the image
    logging.info("Tokenizing the question...")
    question = question.replace("?", "").strip()                    # remove the question mark (if present) & extra spaces before tokenizing
    tokenized_question = tokenizeQuestion(config, question, device)
    logging.info("Featurizing the image...")
    featurized_img = featurizeImage(config, img_path, device)
    
    # Load the model
    logging.info("Loading the {0} model...".format(config["model"]["name"]))
    model = MultimodalVQAModel(
        num_labels=len(answer_space),
        intermediate_dims=config["model"]["intermediate_dims"],
        dropout=config["model"]["dropout"],
        pretrained_text_name=config["model"]["text_encoder"],
        pretrained_image_name=config["model"]["image_encoder"]
    )
    checkpoint = os.path.join(config["train"]["output_dir"], config["model"]["name"], config["inference"]["checkpoint"], "pytorch_model.bin")
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    
    model.eval()
    
    # Obtain the prediction from the model
    logging.info("Obtaining predictions...")
    input_ids = tokenized_question["input_ids"].to(device)
    token_type_ids = tokenized_question["token_type_ids"].to(device)
    attention_mask = tokenized_question["attention_mask"].to(device)
    pixel_values = featurized_img["pixel_values"].to(device)
    output = model(input_ids, pixel_values, attention_mask, token_type_ids)
    
    # Obtain the answer from the answer space
    preds = output["logits"].argmax(axis=-1).cpu().numpy()
    answer = answer_space[preds[0]]
    print()
    print("**********************************")
    print("Answer:\t{0}".format(answer))
    print("**********************************")

    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', required=True, help="YAML file containing parameters & configs")
    args_parser.add_argument('--img_path', required=True, help="path to image file")
    args_parser.add_argument('--question', required=True, help="question to be asked")
    args = args_parser.parse_args()
    
    main(args.config, args.img_path, args.question)