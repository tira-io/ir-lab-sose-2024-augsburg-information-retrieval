#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random

class BaseDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                  padding='max_length', return_tensors='pt')
        return {k: v.squeeze(0) for k, v in encoding.items()}

class MLMDataset(BaseDataset):
    def __init__(self, texts, tokenizer, max_length=512, mlm_probability=0.15):
        super().__init__(texts, tokenizer, max_length)
        self.mlm_probability = mlm_probability

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        inputs, labels = self.mask_tokens(item['input_ids'])
        return {'input_ids': inputs, 'attention_mask': item['attention_mask'], 'labels': labels}

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # only compute loss on masked tokens

        # 80% ->  replace with tokenizer.mask_token [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% -> replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest (10%) -> keep the masked input tokens unchanged
        return inputs, labels


class ContrastiveDataset(BaseDataset):
    def __init__(self, texts, tokenizer, max_length=512):
        super().__init__(texts, tokenizer, max_length)

    def __getitem__(self, idx):
        anchor = super().__getitem__(idx)
        positive = self.get_positive_sample(idx)
        negative = self.get_negative_sample(idx)
        return { 'anchor': anchor, 'positive': positive, 'negative': negative }

    def get_positive_sample(self, idx):
        # This is a placeholder. In a real scenario, you'd have a way to get a semantically similar text.
        # For now, we'll just use the same text with some random word replacements.
        # TODO: GET REAL SIMILAR TEXT!
        text = self.texts[idx]
        words = text.split()
        num_replacements = max(1, len(words) // 10)
        for _ in range(num_replacements):
            i = random.randint(0, len(words) - 1)
            words[i] = random.choice(words)
        positive_text = ' '.join(words)
        return super().__getitem__(self.texts.index(positive_text))

    def get_negative_sample(self, idx):
        # Get a random text that's not the current one # FIXME: needs to be not similar!
        negative_idx = random.choice([i for i in range(len(self.texts)) if i != idx])
        return super().__getitem__(negative_idx)

def get_dataloader(tokenizer, texts, mode, batch_size=4, shuffle=True, **dataset_params):
    if mode == "mlm":
        dataset = MLMDataset(tokenizer=tokenizer, texts=texts, **dataset_params)
    elif mode == "contrastive":
        dataset = ContrastiveDataset(tokenizer=tokenizer, texts=texts, **dataset_params)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


## Usage:
#tokenizer = AutoTokenizer.from_pretrained("prajjwa1/tiny-bert")
#texts = ["Example passage 1", "Example passage 2"] # TODO: data.py
#
#mlm_dataloader = get_dataloader(tokenizer, "mlm", mlm_probability=0.1)
#contrastive_dataloader = get_dataloader(tokenizer, "contrastive")
