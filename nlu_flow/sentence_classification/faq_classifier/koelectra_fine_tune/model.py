from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import ElectraModel, ElectraTokenizer

from nlu_flow.utils import meta_db_client

from tqdm import tqdm
from argparse import Namespace, ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

import os
import multiprocessing
import random

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

            self.utterances.append(data["question"])
            self.labels.append(data["faq_intent"])

        '''
        scenario_data = meta_db_client.get("nlu-intent-entity-utterances")
        for data in tqdm(random.choices(scenario_data, k=len(self.utterances)), desc=f"collecting scenario data ... "):
            self.utterances.append(data["utterance"])
            self.labels.append('scenario')
        '''

        label_statistics = {}

        # organize label_dict
        for label in self.labels:
            self.label_idx[label] = len(self.label_idx)
            label_statistics[label] = label_statistics.get(label, 0) + 1

        for k,v in self.label_idx.items():
            self.idx_label[v] = k

        label_staistics = {k: v for k, v in sorted(label_statistics.items(), key=lambda item: item[1])}
        for k, v in label_statistics.items():
            print (f'{k}\t{v}')

    def tokenize(self, text, pad=True, max_length=20):
        tokens = self.tokenizer.encode(text)     
        if pad:
            tokens += ([self.tokenizer.pad_token_id] * max_length)

        return tokens[:max_length]

    def __getitem__(self, i):
        tokens = self.tokenize(self.utterances[i])
        tokens = torch.LongTensor(tokens)
        label = torch.tensor(self.label_idx[self.labels[i]])

        return tokens, label

    def __len__(self):
        return len(self.utterances)

class KoelectraFAQClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.embedding_net = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        self.loss_fn = nn.CrossEntropyLoss()

        self.hparams = hparams
        self.lr = self.hparams.lr
        self.batch_size = self.hparams.batch_size

        self.dataset = FAQDataset()
        print(f'total dataset num:{len(self.dataset)}')
        self.feature_layer = nn.Linear(self.embedding_net.config.hidden_size, len(self.dataset.label_idx))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def forward(self, x):
        x = self.embedding_net(x)[0][:,0,:] # tuple - first token
        x = self.feature_layer(x)

        return x

    def inference(self, text, top_k=3):
        tokens = self.dataset.tokenize(text)
        pred = self.forward(torch.LongTensor(tokens).unsqueeze(0))

        topk = torch.topk(pred, top_k)[1].numpy()
        labels = []
        for each_pred in topk:
            labels.append(self.dataset.idx_label[each_pred])

        return [{'confidence': k, 'faq_intent': label} for k, label in zip(topk, labels)]

    def training_step(self, batch, batch_idx):
        tokens, label = batch

        pred = self.forward(tokens)
        loss = self.loss_fn(pred, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor':'train_loss'}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--batch_size', default=128)
    args = parser.parse_args()

    model = KoelectraFAQClassifier(args)

    checkpoint_callback = ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min', filepath='koelectra_faq_classifier.ckpt')
  
    trainer = Trainer(callbacks=[EarlyStopping(monitor='train_loss')], checkpoint_callback=checkpoint_callback)
    trainer.fit(model)
