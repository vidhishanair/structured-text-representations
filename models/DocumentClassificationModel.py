import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention


class DocumentClassificationModel(nn.Module):
    def __init__(self, device, vocab_size, token_emb_size, sent_hidden_size, doc_hidden_size, sent_num_layers, doc_num_layers, sem_dim_size, pretrained=None, dropout=0.5):
        super(DocumentClassificationModel, self).__init__()
        self.device = device
        self.word_lookup = nn.Embedding(vocab_size, token_emb_size)
        #self.drop = nn.Dropout(dropout)
        #self.emb_drop = nn.Dropout(dropout)
        if pretrained is not None:
            self.word_lookup.weight.data.copy_(torch.from_numpy(pretrained))
        del pretrained
        self.sentence_encoder = BiLSTMEncoder(sent_hidden_size, token_emb_size, sent_num_layers)
        self.document_encoder = BiLSTMEncoder(doc_hidden_size, sem_dim_size, doc_num_layers)

        self.sentence_structure_att = StructuredAttention(device, sem_dim_size, sent_hidden_size)
        self.document_structure_att = StructuredAttention(device, sem_dim_size, doc_hidden_size)

        self.linear_out = nn.Linear(sem_dim_size, 5)


    def forward(self, input):
        """

        :param input: batch*document_size*sent_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        batch_size, sent_size, token_size = input['token_idxs'].size()
        # sent_l = input['sent_l']
        # sent_mask = input['mask_tokens']
        input = self.word_lookup(input['token_idxs'])
        #input = self.emb_drop(input)
        #reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))

        #BiLSTM
        encoded_sentences, hidden = self.sentence_encoder.forward(input)

        #Structure ATT
        encoded_sentences = self.sentence_structure_att.forward(encoded_sentences)

        #Reshape and max pool
        encoded_sentences = encoded_sentences.contiguous().view(batch_size, sent_size, token_size, encoded_sentences.size(2))
        encoded_sentences = encoded_sentences.max(dim=2)[0] # Batch * sent * dim

        #Doc BiLSTM
        encoded_documents, hidden = self.document_encoder.forward(encoded_sentences)
        #structure Att
        encoded_documents = self.document_structure_att.forward(encoded_documents)
        #Max Pool
        encoded_documents = encoded_documents.max(dim=1)[0]

        #Linear for output
        output = self.linear_out(encoded_documents)

        return output
