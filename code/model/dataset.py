import torch
from torch.utils.data import Dataset, DataLoader
import json

class TextDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []

        print ("load data")
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print ("success")
        
        from tqdm import tqdm
        for item in tqdm(data):
            self.texts.append(item) 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        inputs = {key: val.squeeze() for key, val in inputs.items()}

        return inputs

