import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention
import time


class DocumentClassificationModel(nn.Module):
    def __init__(self, device, vocab_size, token_emb_size, sent_hidden_size, doc_hidden_size, sent_num_layers, doc_num_layers, sem_dim_size, pretrained=None, dropout=0.5, bidirectional=True):
        super(DocumentClassificationModel, self).__init__()
        self.device = device
        self.word_lookup = nn.Embedding(vocab_size, token_emb_size)
        #self.word_lookup.weight.requires_grad = False
        #self.drop = nn.Dropout(dropout)
        #self.emb_drop = nn.Dropout(dropout)
        if pretrained is not None:
            self.word_lookup.weight.data.copy_(torch.from_numpy(pretrained))
        del pretrained
        if bidirectional:
            self.sem_dim_size = 2*sem_dim_size
            self.sent_hidden_size = 2*sent_hidden_size
            self.doc_hidden_size = 2*doc_hidden_size
        else:
            self.sem_dim_size = sem_dim_size
            self.sent_hidden_size = sent_hidden_size
            self.doc_hidden_size = doc_hidden_size
        self.sentence_encoder = BiLSTMEncoder(self.sent_hidden_size, token_emb_size, sent_num_layers, dropout, bidirectional)
        self.document_encoder = BiLSTMEncoder(self.doc_hidden_size, self.sem_dim_size, doc_num_layers, dropout, bidirectional)

        self.sentence_structure_att = StructuredAttention(device, self.sem_dim_size, self.sent_hidden_size, bidirectional)
        self.document_structure_att = StructuredAttention(device, self.sem_dim_size, self.doc_hidden_size, bidirectional)

        self.pre_lin1 = nn.Linear(self.sem_dim_size, self.sem_dim_size)
        self.pre_lin2 = nn.Linear(self.sem_dim_size, self.sem_dim_size)

        self.linear_out = nn.Linear(self.sem_dim_size, 5)


    def forward(self, input):
        """

        :param input: batch*document_size*sent_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        start_time = time.time()
        
        batch_size, sent_size, token_size = input['token_idxs'].size()
        # sent_l = input['sent_l']

        tokens_mask = input['mask_tokens']
        sent_mask = input['mask_sents']

        input = self.word_lookup(input['token_idxs'])
        #input = self.emb_drop(input)

        #reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))

        #BiLSTM
        encoded_sentences, hidden = self.sentence_encoder.forward(input)
#         print("--- %s seconds for BiLSTM ---" % (time.time() - start_time))
        
        
        #Structure ATT
        encoded_sentences = self.sentence_structure_att.forward(encoded_sentences)
#         print("--- %s seconds for Structured Attention for sentence ---" % (time.time() - start_time))
        
        #Reshape and max pool
        encoded_sentences = encoded_sentences.contiguous().view(batch_size, sent_size, token_size, encoded_sentences.size(2))
        encoded_sentences = encoded_sentences + ((tokens_mask-1)*9999).unsqueeze(3).repeat(1,1,1,encoded_sentences.size(3))
        encoded_sentences = encoded_sentences.max(dim=2)[0] # Batch * sent * dim
#         print("--- %s seconds for Reshaping and pooling ---" % (time.time() - start_time))
        
        #Doc BiLSTM
        encoded_documents, hidden = self.document_encoder.forward(encoded_sentences)
#         print("--- %s seconds for BiLSTM for document ---" % (time.time() - start_time))
        #structure Att
        
        encoded_documents = self.document_structure_att.forward(encoded_documents)
#         print("--- %s seconds for Structured Attention document ---" % (time.time() - start_time)) 
        
        #print(encoded_documents.size())
        #Max Pool
        encoded_documents = encoded_documents + ((sent_mask-1)*9999).unsqueeze(2).repeat(1,1,encoded_documents.size(2))
        encoded_documents = encoded_documents.max(dim=1)[0]

        encoded_documents = F.relu(self.pre_lin1(encoded_documents))
        encoded_documents = F.relu(self.pre_lin2(encoded_documents))

        #Linear for output
        output = self.linear_out(encoded_documents)
        
#         print("--- %s seconds final ---" % (time.time() - start_time))
        
        return output
