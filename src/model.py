from typing import List, Dict, Optional
import torch
from torch import nn
from transformers import AutoModel


class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int,
            intermediate_dims: int,
            dropout: float,
            pretrained_text_name: str,
            pretrained_image_name: str):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(intermediate_dims, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }

        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


def createMultimodalModelForVQA(config: Dict, answer_space: List[str]) -> MultimodalVQAModel:
    model = MultimodalVQAModel(
        num_labels=len(answer_space),
        intermediate_dims=config["model"]["intermediate_dims"],
        dropout=config["model"]["dropout"],
        pretrained_text_name=config["model"]["text_encoder"],
        pretrained_image_name=config["model"]["image_encoder"]
    )

    return model