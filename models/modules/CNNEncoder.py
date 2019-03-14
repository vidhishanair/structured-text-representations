import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, device, hidden_size, input_size, num_layers, dropout=0.5, bidirectional=True):
        super(CNNEncoder, self).__init__()
        # if bidirectional:
        #     hidden_size = hidden_size//2
        self.device = device
        filter_size = 5
        sequence_len = 30
        self.strides = (1, 1)
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=(filter_size, 1),
                               stride=self.strides,
                               bias=True)
        self.relu1 = nn.ReLU()
        pooling_window_size = sequence_len - filter_size + 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, pooling_window_size, 1, 1), stride=(1, 1, 1, 1))


    def forward(self, input, seq_len):
        """
        Returns CNN encoded sequence
        :param input: batch*document_size*sent_size*emb_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        # print(seq_len)
        # pack = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True)
        output, hidden = self.bilstm(input)
        # output, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)

        return output, hidden

    def forward_packed(self, input, seq_len):
        # Sort by length (keep idx)
        batch_size, seq_len, dim = input.size()
        input = input.transpose(0,1).transpose(2,1)
        h_conv = self.conv1(input)
        h_relu = self.relu1(h_conv)
        h_max = self.maxpool1(h_relu)
        h_flat = h_max.view(-1, self.hidden_size)
        h_flat = h_flat.view(batch_size, seq_len, self.hidden_size)
        return h_flat, None

