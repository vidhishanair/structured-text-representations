import torch.nn as nn
import torch
import torch.nn.functional as F



class StructuredAttention(nn.Module):
    def __init__(self, device, sem_dim_size, sent_hiddent_size):
        super(StructuredAttention, self).__init__()
        self.device = device
        self.sem_dim_size = sem_dim_size
        self.att_dim = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size
        self.rep_dim = sem_dim_size
        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=True)
        self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=True)

        self.exparam = nn.Parameter(torch.Tensor(1,1,self.sem_dim_size))
        self.fzlinear = nn.Linear(2*self.sem_dim_size, self.sem_dim_size)

    def forward(self, input): #batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        sem_v = input[:,:,:self.sem_dim_size]
        str_v = input[:,:,self.sem_dim_size:]

        tp = F.tanh(self.tp_linear(str_v)) # b*s, token, h1
        tc = F.tanh(self.tc_linear(str_v)) # b*s, token, h1
        tp = tp.unsqueeze(2).expand(tp.size(0), tp.size(1), tp.size(1), tp.size(2)).contiguous()
        tc = tc.unsqueeze(2).expand(tc.size(0), tc.size(1), tc.size(1), tc.size(2)).contiguous()

        f_ij = self.bilinear(tp, tc).squeeze() # b*s, token , token
        f_i = torch.exp(self.fi_linear(str_v)).squeeze()  # b*s, token

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).to(self.device)
        A_ij = torch.exp(f_ij)*mask

        del mask, tp, tc, f_ij

        tmp = torch.sum(A_ij, dim=1)
        res = torch.zeros(batch_size, token_size, token_size).to(self.device)
        #tmp = torch.stack([torch.diag(t) for t in tmp])
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res   #A_ij has 0s as diagonals

        del res, tmp

        L_ij_bar = L_ij
        L_ij_bar[:,0,:] = f_i

        #No batch inverse
        LLinv = torch.stack([torch.inverse(li) for li in L_ij_bar])
        #LLinv = b_inv(L_ij_bar, self.device).contiguous()



        d0 = f_i * LLinv[:,:,0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (LLinv_diag * A_ij.transpose(1,2)).transpose(1,2)
        tmp2 = A_ij * LLinv_diag.transpose(1,2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size-1)
        temp22 = torch.ones(batch_size, token_size-1, token_size)

        mask1 = torch.cat([temp11,temp12],2).to(self.device)
        mask2 = torch.cat([temp21,temp22],1).to(self.device)

        dx = mask1 * tmp1 - mask2 * tmp2

        del tmp1, tmp2, temp11, temp12, temp21, temp22, mask1, mask2, L_ij_bar, LLinv_diag, A_ij, f_i, LLinv, L_ij

        d = torch.cat([d0.unsqueeze(1), dx], dim = 1)
        df = d.transpose(1,2)

        ssr = torch.cat([self.exparam.repeat(batch_size,1,1), sem_v], 1)
        pinp = torch.bmm(df, ssr)

        cinp = torch.bmm(dx, sem_v)

        finp = torch.cat([sem_v, pinp],dim = 2)
        output = F.relu(self.fzlinear(finp))

        return output

def b_inv(b_mat, device):
    #eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).to(device)
    # eye = torch.eye(b_mat.size(1), b_mat.size(2)).unsqueeze(0).expand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).to(self.device)
    eye = torch.rand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).to(device)
    #b_inv = torch.rand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).to(device)
    b_inv, _ = torch.gesv(eye, b_mat)
    del eye
    return b_inv
