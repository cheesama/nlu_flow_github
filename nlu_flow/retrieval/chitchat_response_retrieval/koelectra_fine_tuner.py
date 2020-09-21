from transformers import ElectraModel, ElectraTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

class KoelectraQAFineTuner(nn.Module):
    def __init__(
        self,
        max_len=64,
        feature_dim=128
    ):
        super(KoelectraQAFineTuner, self).__init__()

        self.question_net = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        self.answer_net = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")

        self.question_feature = nn.Linear(max_len, feature_dim)
        self.question_feature.bias.data.zero_()
        nn.init.xavier_uniform_(self.question_feature.weight)

        self.answer_feature = nn.Linear(max_len, feature_dim)
        self.answer_feature.bias.data.zero_()
        nn.init.xavier_uniform_(self.answer_feature.weight)

    def forward(self, q, a, label):
        question_features = self.get_question_feature(q)
        answer_features = self.get_answer_feature(a)

        sim_value = torch.mm(question_features, answer_features.transpose(1,0))

        label = label.unsqueeze(1)
        pos_pair = (label == label.transpose(1,0)).float()
        neg_pair = (label != label.transpose(1,0)).float()

        #ignore negative similarity of negative pair
        #neg_loss = (neg_pair * sim_value).clamp(min=0.0).mean()
        neg_loss = (1 + (neg_pair * sim_value)).mean()
        pos_loss = (1 - (pos_pair * sim_value)).mean()
        loss = neg_loss + pos_loss

        return loss, pos_loss, neg_loss

    def get_question_feature(self, q):
        question_features = F.normalize(self.question_feature(self.question_net(q)[0].mean(2)), dim=1)
        return question_features

    def get_answer_feature(self, a):
        answer_features = F.normalize(self.answer_feature(self.answer_net(a)[0].mean(2)), dim=1)
        return answer_features
