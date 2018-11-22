import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np


class BiLSTMEncoder(nn.Module):
    def __init__(self, device, hidden_size, input_size, num_layers, dropout=0.5, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        if bidirectional:
            hidden_size = hidden_size//2
        self.device = device
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)

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

    def forward_packed(self, input, seq_len):
        # Sort by length (keep idx)
        seq_len = np.array(seq_len)
        sent_len, idx_sort = np.sort(seq_len)[::-1], np.argsort(-seq_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).to(self.device)
        sent_variable = input.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        #print(seq_len)
        #print(sent_len)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, sent_len.copy(), batch_first=True)
        sent_output, hidden = self.bilstm(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        sent_output = sent_output.index_select(0, idx_unsort)

        del idx_sort, idx_unsort
        return sent_output, hidden
