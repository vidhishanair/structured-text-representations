import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTMEncoder(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, dropout=0.5, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        if bidirectional:
            hidden_size = hidden_size//2
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional, dropout=dropout)

    def forward(self, input):
        """
        Returns BiLSTM encoded sequence
        :param input: batch*document_size*sent_size*emb_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        output, hidden = self.bilstm(input)
        return output, hidden
