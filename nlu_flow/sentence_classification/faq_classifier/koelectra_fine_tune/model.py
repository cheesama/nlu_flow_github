from torch import nn
from torch.utils.data import DataLoader, random_split

from transformers import ElectraModel, ElectraTokenizer

from nlu_flow.utils import meta_db_client

from tqdm import tqdm

import torch
import torch.nn.functional as F

import pytorch_lightning as pl, Trainer

import os

class FAQDataset(torch.utils.data.Dataset):
    def __init__(self):
        # load dataset
        self.utterances = []
        self.labels = []
        self.label_idx = {}
        self.idx_label = {}
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
        
        ## get synonym data for data augmentation(for FAQ data augmentation)
        synonyms = []
        synonym_data = meta_db_client.get("meta-entities")
        for data in tqdm(
            synonym_data, desc=f"collecting synonym data for data augmentation ..."
        ):
            if type(data) != dict:
                print(f"check data type : {data}")
                continue

            synonyms.append([each_synonym.get("synonym") for each_synonym in data.get("meta_synonyms")] + [data.get("Entity_Value")])

        faq_data = meta_db_client.get("nlu-faq-questions")
        for data in tqdm(faq_data, desc=f"collecting faq data ... "):
            if data["faq_intent"] is None or len(data["faq_intent"]) < 2:
                print(f"check data! : {data}")
                continue

            target_utterance = data["question"]

            # check synonym is included
            for synonym_list in synonyms:
                for i, prev_value in enumerate(synonym_list):
                    if prev_value in target_utterance:
                        for j, post_value in enumerate(synonym_list):
                            if i == j:
                                continue
                            self.utterances.append(target_utterance.replace(prev_value, post_value))
                            self.labels.append(data["faq_intent"])

                        break

            self.utterances.append(data["question"])
            self.labels.append(data["faq_intent"])

        scenario_data = meta_db_client.get("nlu-intent-entity-utterances")
        for data in tqdm(scenario_data, desc=f"collecting scenario data ... "):
            self.utterances.append(data["utterance"])
            self.labels.append('시나리오')

        # organize label_dict
        for label in self.labels:
            self.label_idx[label] = len(self.label_idx)
        for k,v in self.label_idx:
            self.idx_label[v] = k

    def tokenize(self, text, pad=True, max_length=20):
        tokens = self.tokenizer.encode(text)     
        if pad:
            tokens += ([self.tokenizer.pad_token_id] * max_length)

        return tokens[:max_length]

    def __getitem__(self, i):
        tokens = self.tokenize(self.utterances[i])
        tokens = torch.LongTensor(tokens)
        label = torch.LongTensor(self.label_idx[self.labels[i]])

        return tokens, label

    def __len__(self):
        return len(self.utterances)

class KoelectraFAQClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, batch_size=64):
        super().__init__()
        self.embedding_net = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        
        self.dataset = FAQDataset()
        self.feature_layer = nn.Linear(self.embedding_net.config.hidden_size, len(self.dataset.self.label_idx))

        self.lr = lr
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def forward(self, x):
        x = self.embedding_net(x)[0][0] # tuple - first token
        x = self.feature_layer(x)

        return x

    def inference(self, text):
        tokens = self.dataset.tokenize(text)
        pred = self.forward(torch.LongTensor(tokens).unsqueeze(0))

        return pred.argmax(1)[0], self.dataset.idx_label[pred.argmax(1)[0]]

    def training_step(self, batch, batch_idx):
        tokens, label = batch

        pred = self.forward(tokens)
        loss = F.cross_entropy(pred, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

model = KoelectraFAQClassifier()
  
trainer = Trainer()
trainer.fit(model)
    
        

        
