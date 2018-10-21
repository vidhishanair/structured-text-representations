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

    def forward(self, input, seq_len):
        """
        Returns BiLSTM encoded sequence
        :param input: batch*document_size*sent_size*emb_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        # print(seq_len)
        # pack = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True)
        output, hidden = self.bilstm(input)
        # output, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)

        return output, hidden
