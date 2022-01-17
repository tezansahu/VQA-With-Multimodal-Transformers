import os
from typing import Dict, Text
from datasets import load_dataset

def loadDaquarDataset(config: Dict) -> Dict:
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["eval_dataset"])
        }
    )

    with open(os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"])) as f:
        answer_space = f.read().splitlines()

    
    dataset = dataset.map(
        lambda examples: {
            'label': [
                answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
                for ans in examples[config["data"]["answer_col"]]
            ]
        },
        batched=True
    )

    return {
        "dataset": dataset,
        "answer_space": answer_space
    }