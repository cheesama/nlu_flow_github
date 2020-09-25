from transformers import ElectraModel, ElectraTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoelectraQAFineTuner(nn.Module):
    def __init__(self, class_num):
        super(KoelectraQAFineTuner, self).__init__()

        self.embedding_net = ElectraModel.from_pretrained(
            "monologg/koelectra-small-v2-discriminator"
        )

        self.feature = nn.Linear(self.embedding_net.config.hidden_size, class_num)

    def forward(self, text):
        feature = self.embedding_net(text)[0][:, 0, :]
        pred = self.feature(feature)

        return pred
