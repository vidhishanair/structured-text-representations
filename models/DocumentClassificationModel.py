import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention
import itertools
import time


class DocumentClassificationModel(nn.Module):
    def __init__(self, device, vocab_size, token_emb_size, sent_hidden_size, doc_hidden_size, sent_num_layers, doc_num_layers, sem_dim_size, pretrained=None, dropout=0.5, bidirectional=True, py_version='nightly'):
        super(DocumentClassificationModel, self).__init__()
        self.device = device
        self.word_lookup = nn.Embedding(vocab_size, token_emb_size)
        #self.word_lookup.weight.requires_grad = False
        self.drop = nn.Dropout(dropout)
        if pretrained is not None:
            self.word_lookup.weight.data.copy_(torch.from_numpy(pretrained))

        if bidirectional:
            self.sem_dim_size = 2*sem_dim_size
            self.sent_hidden_size = 2*sent_hidden_size
            self.doc_hidden_size = 2*doc_hidden_size
        else:
            self.sem_dim_size = sem_dim_size
            self.sent_hidden_size = sent_hidden_size
            self.doc_hidden_size = doc_hidden_size

        self.sentence_encoder = BiLSTMEncoder(device, self.sent_hidden_size, token_emb_size, sent_num_layers, dropout, bidirectional)
        self.document_encoder = BiLSTMEncoder(device, self.doc_hidden_size, self.sem_dim_size, doc_num_layers, dropout, bidirectional)

        self.sentence_structure_att = StructuredAttention(device, self.sem_dim_size, self.sent_hidden_size, bidirectional, py_version)
        self.document_structure_att = StructuredAttention(device, self.sem_dim_size, self.doc_hidden_size, bidirectional, py_version)

        self.pre_lin1 = nn.Linear(self.sem_dim_size, self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.pre_lin1.weight)
        nn.init.constant_(self.pre_lin1.bias, 0)
        self.pre_lin2 = nn.Linear(self.sem_dim_size, self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.pre_lin2.weight)
        nn.init.constant_(self.pre_lin2.bias, 0)
        self.linear_out = nn.Linear(self.sem_dim_size, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)


    def forward(self, input):
        """

        :param input: batch*document_size*sent_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        start_time = time.time()
        
        batch_size, sent_size, token_size = input['token_idxs'].size()
        sent_l = input['sent_l']
        doc_l = input['doc_l']

        tokens_mask = input['mask_tokens']
        sent_mask = input['mask_sents']

        input = self.word_lookup(input['token_idxs'])
        input = self.drop(input)

        #reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))
        sent_l = list(itertools.chain.from_iterable(sent_l))


        #BiLSTM
        #print(sent_l)
        encoded_sentences, hidden = self.sentence_encoder.forward_packed(input, sent_l)

        mask = tokens_mask.view(tokens_mask.size(0)*tokens_mask.size(1), tokens_mask.size(2)).unsqueeze(2).repeat(1,1,encoded_sentences.size(2))
        #print(encoded_sentences.size())
        #print(mask.size())
        encoded_sentences = encoded_sentences * mask

        #Structure ATT
        encoded_sentences, sent_attention_matrix = self.sentence_structure_att.forward(encoded_sentences)

        #Reshape and max pool
        encoded_sentences = encoded_sentences.contiguous().view(batch_size, sent_size, token_size, encoded_sentences.size(2))
        encoded_sentences = encoded_sentences + ((tokens_mask-1)*999).unsqueeze(3).repeat(1,1,1,encoded_sentences.size(3))
        encoded_sentences = encoded_sentences.max(dim=2)[0] # Batch * sent * dim

        #Doc BiLSTM
        encoded_documents, hidden = self.document_encoder.forward_packed(encoded_sentences, doc_l)
        mask = sent_mask.unsqueeze(2).repeat(1,1,encoded_documents.size(2))
        encoded_documents = encoded_documents * mask

        #structure Att
        encoded_documents, doc_attention_matrix = self.document_structure_att.forward(encoded_documents)

        #Max Pool
        encoded_documents = encoded_documents + ((sent_mask-1)*999).unsqueeze(2).repeat(1,1,encoded_documents.size(2))
        encoded_documents = encoded_documents.max(dim=1)[0]

        #encoded_documents = self.drop(encoded_documents)
        #encoded_documents = F.relu(self.pre_lin1(encoded_documents))
        #encoded_documents = self.drop(encoded_documents)
        #encoded_documents = F.relu(self.pre_lin2(encoded_documents))

        #Linear for output
        output = self.linear_out(encoded_documents)

        return output, sent_attention_matrix, doc_attention_matrix
