from Korpora import Korpora, KoreanChatbotKorpus
from tqdm import tqdm

from embedding_transformer import EmbeddingTransformer

from nlu_flow.preprocessor.text_preprocessor import normalize
from nlu_flow.utils.kor_char_tokenizer import KorCharTokenizer

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn

import os, sys
import multiprocessing

# downlad dataset
chatbot_corpus = KoreanChatbotKorpus()

tokenizer = KorCharTokenizer()

# prepare torch dataset
class ChatbotKorpusDataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, tokenizer):
        assert len(questions) == len(answers)

        self.tokenizer = tokenizer
        self.dataset = []

        for i, question in tqdm(enumerate(questions), desc="preparing data ..."):
            question_tokens = self.tokenizer.tokenize(questions[i])
            answer_tokens = self.tokenizer.tokenize(answers[i])
            self.dataset.append((question_tokens, answer_tokens))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question_tensor = torch.LongTensor(self.tokenizer.tokenize(self.dataset[idx][0]))
        answer_tensor = torch.LongTensor(self.tokenizer.tokenize(self.dataset[idx][1]))

        return question_tensor, answer_tensor

questions = []
answers = []

for qa in chatbot_corpus.train:
    questions.append(qa.text)
    answers.append(qa.pair)

train_dataset = ChatbotKorpusDataset(questions, answers, tokenizer)
train_loader = DataLoader(
    train_dataset, batch_size=64, num_workers=multiprocessing.cpu_count()
)

# model definition
model = EmbeddingTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=tokenizer.get_seq_len(),
    pad_token_id=tokenizer.get_pad_token_id(),
)

# train model
n_epochs = 10
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, n_epochs + 1):
    model.train()

    progress = tqdm(enumerate(train_loader), leave=False)
    for batch_idx, (question, answer) in progress:
        optimizer.zero_grad()
        pred = model(question)

        loss = loss_fn(pred.transpose(1,2), answer)
        loss.backward()
        optimizer.step()

        progress.set_description(f'training model, epoch:{epoch}, loss:{loss.item()}')

    torch.save(model.state_dict(), 'transformer_chitchat_response_model.modeldict')
