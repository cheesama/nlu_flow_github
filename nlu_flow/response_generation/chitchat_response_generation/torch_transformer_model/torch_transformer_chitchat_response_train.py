from Korpora import Korpora, KoreanChatbotKorpus
from tqdm import tqdm

from embedding_transformer import EmbeddingTransformer

from nlu_flow.preprocessor.text_preprocessor import normalize
from nlu_flow.utils.kor_char_tokenizer import KorCharTokenizer

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

import os, sys
import multiprocessing
import argparse

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

        #print (f'question_tokens: {self.dataset[idx][0]}')
        #print (f'answer_tokens: {self.dataset[idx][1]}')

        return question_tensor, answer_tensor

questions = []
answers = []

for qa in chatbot_corpus.train:
    questions.append(qa.text)
    answers.append(qa.pair)

train_dataset = ChatbotKorpusDataset(questions, answers, tokenizer)

def train_model(batch_size=128, n_epochs=20, lr=0.0001):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()
    )

    # model definition
    model = EmbeddingTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=tokenizer.get_seq_len(),
        pad_token_id=tokenizer.get_pad_token_id(),
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # train model
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())

    writer = SummaryWriter(log_dir=f'runs/epochs:{n_epochs}_lr:{lr}')
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()

        progress = tqdm(enumerate(train_loader), leave=False)
        for batch_idx, (question, answer) in progress:
            if torch.cuda.is_available():
                question = question.cuda()
                answer = answer.cuda()

            optimizer.zero_grad()
            pred = model(question)

            loss = loss_fn(pred.transpose(1,2), answer)
            loss.backward()
            optimizer.step()

            progress.set_description(f'training model, epoch:{epoch}, loss:{loss.cpu().item()}')
            writer.add_scalar('Loss/train', loss.cpu().item(), global_step)
            global_step += 1

        torch.save(model.state_dict(), 'transformer_chitchat_response_model.modeldict')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--n_epochs', default=128)
    parser.add_argument('--lr', default=0.0001)
    args = parser.parse_args()

    train_model(args.batch_size, args.n_epochs, args.lr)
