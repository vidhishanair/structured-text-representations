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
        self.drop = nn.Dropout(dropout)
        self.emb_drop = nn.Dropout(dropout)
        if pretrained is not None:
            self.word_lookup.weight.data.copy_(torch.from_numpy(pretrained))
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
        sent = input['token_idxs']
        sent_l = input['sent_l']
        sent_mask = input['mask_tokens']
        input = self.word_lookup(sent)
        input = self.emb_drop(input)
        reshaped_input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))
        encoded_sentences, hidden = self.sentence_encoder.forward(reshaped_input)

        structured_encoded_sentences = self.sentence_structure_att.forward(encoded_sentences)
        """
        TODO
        1. Structured Attention
        2. Pooling
        3. Document Encoder
        4. Structured Attention
        5. Pooling
        6. Prediction
        """
        encoded_sentences = structured_encoded_sentences.contiguous().view(input.size(0), input.size(1), input.size(2), structured_encoded_sentences.size(2))


        encoded_sentences = encoded_sentences.max(dim=2)[0] # Batch * sent * dim
        encoded_documents, hidden = self.document_encoder.forward(encoded_sentences)
        structured_encoded_documents = self.document_structure_att.forward(encoded_documents)
        encoded_documents = structured_encoded_documents.max(dim=1)[0]
        output = self.linear_out(encoded_documents)
        return output
