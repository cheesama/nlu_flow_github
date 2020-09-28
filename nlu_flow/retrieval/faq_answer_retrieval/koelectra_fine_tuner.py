from transformers import ElectraModel, ElectraTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoelectraQAFineTuner(nn.Module):
    def __init__(self, margin=0.5):
        super(KoelectraQAFineTuner, self).__init__()

        self.question_net = ElectraModel.from_pretrained(
            "monologg/koelectra-small-v2-discriminator"
        )
        self.answer_net = ElectraModel.from_pretrained(
            "monologg/koelectra-small-v2-discriminator"
        )

        self.margin = margin

    def forward(self, q, a):
        question_features = self.get_question_feature(q)
        answer_features = self.get_answer_feature(a)

        """
        sim_value = torch.mm(question_features, answer_features.transpose(1,0))
        label = label.unsqueeze(1)
        pair_info = (label == label.transpose(1,0)).float()
        loss = self.margin + ((1 - pair_info) * F.relu(self.margin + sim_value) + (-1.0 * pair_info * sim_value)) * 0.5

        return loss.mean()
        """
        return question_features, answer_features

    def get_question_feature(self, q):
        question_features = F.normalize(self.question_net(q)[0][:, 0, :], dim=1)
        return question_features

    def get_answer_feature(self, a):
        answer_features = F.normalize(self.answer_net(a)[0][:, 0, :], dim=1)
        return answer_features
