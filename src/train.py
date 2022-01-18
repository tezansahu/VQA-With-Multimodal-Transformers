import os
import shutil
from typing import Dict
from transformers import TrainingArguments, Trainer, logging

def setTrainingArgs(config: Dict, device) -> TrainingArguments:
    training_args = config["train"]
    if device.type == 'cuda':
        training_args["fp16"] = True
        
    return TrainingArguments(**training_args)

def trainMultimodalModelForVQA(config, device, dataset, collator, model, compute_metrics):
    training_args = setTrainingArgs(config, device)
    training_args.output_dir = os.path.join(training_args.output_dir, config["model"]["name"])
    
    if os.path.isdir(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)

    multi_trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()
    
    return train_multi_metrics, eval_multi_metrics
