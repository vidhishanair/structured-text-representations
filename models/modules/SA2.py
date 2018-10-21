import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class StructuredAttention(nn.Module):

    def __init__(self, device, input_dim, att_dim, bidirec=True):
        super(StructuredAttention, self).__init__()

        self.bidirec = bidirec
        self.device = device
        self.input_dim = 2*input_dim if bidirec else input_dim
        self.att_dim = 2*att_dim if bidirec else att_dim
        self.rep_dim = 2*(input_dim - att_dim) if bidirec else (input_dim - att_dim)

        self.wplinear = nn.Linear(self.att_dim, self.att_dim)
        self.wclinear = nn.Linear(self.att_dim, self.att_dim)

        self.wabilinear = nn.Bilinear(self.att_dim, self.att_dim, 1, bias = False)
        self.wrlinear = nn.Linear(self.att_dim, 1, bias=False)

        self.fzlinear = nn.Linear(2*self.rep_dim, self.rep_dim)
        self.exparam = nn.Parameter(torch.Tensor(1,1,self.rep_dim))

        nn.init.xavier_uniform_(self.exparam)


    def forward(self, input_var, usecuda = True):

        batch_size, num_words, inp_dim = input_var.size()

        assert inp_dim == self.input_dim, "Dimensions Mismatch Expected %s, Got %s" % (self.input_dim, inp_dim)

        if self.bidirec:
            inpe = torch.cat([input_var[:,:,:self.input_dim//2][:,:,:self.rep_dim//2],
                              input_var[:,:,self.input_dim//2:][:,:,:self.rep_dim//2]], dim =2)
            inpd = torch.cat([input_var[:,:,:self.input_dim//2][:,:,self.rep_dim//2:],
                              input_var[:,:,self.input_dim//2:][:,:,self.rep_dim//2:]], dim =2)

        else:
            inpe = input_var[:,:,:self.rep_dim]
            inpd = input_var[:,:,self.rep_dim:]

        tp = F.tanh(self.wplinear(inpd))
        tc = F.tanh(self.wclinear(inpd))

        tpr = tp.repeat(1,1,num_words).view(batch_size,num_words,num_words,self.att_dim) \
            .view(batch_size*num_words*num_words,self.att_dim)
        tcr = tc.repeat(1,num_words,1).view(batch_size,num_words,num_words,self.att_dim) \
            .view(batch_size*num_words*num_words,self.att_dim)

        fij = self.wabilinear(tpr,tcr).view(batch_size, num_words, num_words)

        fr = torch.exp(self.wrlinear(inpd).squeeze(2))

        Aij = torch.exp(fij)
        maskij = 1.-torch.eye(num_words).unsqueeze(0).expand(batch_size, num_words, num_words).to(self.device)

        Aij = Aij*maskij

        Di = torch.sum(Aij, dim = 1)
        Dij = torch.stack([torch.diag(di) for di in Di])
        Lij = -Aij + Dij

        LLij = Lij[:,1:,:]
        Lfr = fr.unsqueeze(1)
        LLxij = torch.cat([Lfr, LLij], dim = 1)

        #Batch Inverse not available in Pytorch
        LLinv = torch.stack([torch.inverse(li) for li in LLxij])

        d0 = fr * LLinv[:,:,0]

        #Batch Diagonalization not available in pytorch
        LLinv_diag = torch.stack([torch.diag(lid) for lid in LLinv]).unsqueeze(2)

        tmp1 = (LLinv_diag * Aij.transpose(1,2)).transpose(1,2)
        tmp2 = Aij * LLinv_diag.transpose(1,2)

        temp11 = torch.zeros(batch_size,num_words,1)
        temp21 = torch.zeros(batch_size,1,num_words)

        temp12 = torch.ones(batch_size,num_words,num_words-1)
        temp22 = torch.ones(batch_size,num_words-1,num_words)

        mask1 = torch.cat([temp11,temp12],2).to(self.device)
        mask2 = torch.cat([temp21,temp22],1).to(self.device)

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim = 1)
        df = d.transpose(1,2)

        ssr = torch.cat([self.exparam.repeat(batch_size,1,1), inpe], 1)
        pinp = torch.bmm(df, ssr)

        cinp = torch.bmm(dx, inpe)

        finp = torch.cat([inpe, pinp],dim = 2)
        output = F.relu(self.fzlinear(finp))

        return output