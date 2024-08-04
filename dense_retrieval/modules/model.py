
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MLMModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.mlm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.mlm_head(sequence_output)
        return prediction_scores

class ContrastiveModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def get_embedding(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)

class FTModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', mode='mlm'):
        super().__init__()
        self.mode = mode
        if mode == 'mlm':
            self.model = MLMModel(model_name)
        elif mode == 'contrastive':
            self.model = ContrastiveModel(model_name)
        else:
            raise ValueError("Mode must be 'mlm' or 'contrastive'")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)