# %%
import math
from typing import Tuple
import numpy as np
import pandas as pd
import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from torch.nn import TransformerEncoder
from torch.nn.modules.container import ModuleList

from torch.utils.data import dataset
import matplotlib.pyplot as plt

from typing import Optional, Any, Union, Callable

device = torch.device('cuda:0')


# %%
xmin, xmax = 0, 2 * np.pi
Nx, Nt = 201, 200

dx = (xmax - xmin) / (Nx)
x = np.linspace(xmin,xmax, Nx + 1)

num_train = 50
num_test = 1


data_solutions = pd.read_csv('data/Training_data' + str(num_train) + '_Nx_' + str(Nx) + '_Nt_' + str(Nt) + '.csv')
data_solutions = np.reshape(data_solutions.to_numpy(), (num_train, -1 ,Nx + 1))
train_data = (torch.tensor(data_solutions, dtype=torch.float)).transpose(0,1)
train_data.shape # shape [seq_len, total_samples, d_model]

data_dt = pd.read_csv('data/D_array' + str(num_train) + '_Nx_' + str(Nx) + '_Nt_' + str(Nt) + '.csv')
data_DT = np.reshape(data_dt.to_numpy(), (num_train, -1, 1))
train_data_dt = (torch.tensor(data_DT, dtype=torch.float)).transpose(0,1)

test_data_dt = np.reshape(2e-3, (1,1,1))
test_data_dt = (torch.tensor(test_data_dt, dtype=torch.float)).transpose(0,1)


test_data_solutions = pd.read_csv('data/test_same_dt.csv')
test_data = np.reshape(test_data_solutions.to_numpy(), (num_test,-1,Nx + 1))
test_data_same = (torch.tensor(test_data, dtype=torch.float)).transpose(0,1)

val_data = test_data_same

# test_data_solutions = pd.read_csv('data/test_different_dt.csv')
# test_data = np.reshape(test_data_solutions.to_numpy(), (num_test,-1,Nx + 1))
# test_data_different = (torch.tensor(test_data, dtype=torch.float)).transpose(0,1)


# %%
# train_data = test_data_same
print('the train data shape = ', train_data.shape)
print('the train data shape = ', train_data_dt.shape)
print('the train data shape = ',test_data_dt.shape)

# %%
multi_step_num = 1

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return (torch.triu(torch.ones(sz, sz), diagonal=1) + torch.tril(torch.ones(sz, sz), diagonal=-multi_step_num)).bool().to(device)
#     return torch.triu(torch.ones(sz, sz), diagonal=1).bool()


def soft_max_mask(sz: int, num_step: int) -> Tensor:
    if num_step == 1:
        coefficients = torch.Tensor([1])
    if num_step == 2:
        coefficients = torch.Tensor([3/2, -1/2])
    if num_step == 3:
        coefficients = torch.Tensor([23/12, -16/12, 5/12])
    if num_step == 4:
        coefficients = torch.Tensor([55/24, -59/24, 37/24, -9/24])
    
    K = torch.zeros((sz, sz))
    for iii in range(num_step):
        K += coefficients[iii] * torch.diag(torch.ones(sz - iii ,1).squeeze(), diagonal = -iii)
    for iii in range(num_step-1):
        if iii == 0:
                coefficients = torch.Tensor([1])
        if iii == 1:
            coefficients = torch.Tensor([3/2, -1/2])
        if iii == 2:
            coefficients = torch.Tensor([23/12, -16/12, 5/12])
        if iii == 3:
            coefficients = torch.Tensor([55/24, -59/24, 37/24, -9/24])
        coefficients
        for jjj in range(iii+1):
            K[iii, jjj] = coefficients[iii - jjj]
    return K.to(device)


def generate_Dspace(sz:int, a: int, b: int) -> Tensor:
    Identity = torch.diag(torch.ones(sz,1).squeeze(), diagonal = 0)
    D_space = Identity
    if a > 0:
        D_space += np.roll(Identity,1,axis = 0)
    if a > 1:
        D_space += np.roll(Identity,2,axis = 0)
    if a > 2:
        D_space += np.roll(Identity,3,axis = 0)
        
    if b > 0:
        D_space += np.roll(Identity,-1,axis = 0)
    if b > 1:
        D_space += np.roll(Identity,-2,axis = 0)
    
    
    return D_space.to(device)

# %%
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiheadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        # self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)
        nn.init.normal_(self.q_net.weight, mean=0., std=0.01)
        nn.init.normal_(self.k_net.weight, mean=0., std=0.01)
        

        self.Wv = nn.Parameter(torch.normal(0,0.01, (n_head * d_head, d_model)), requires_grad=True)
        self.Wv2 = nn.Parameter(torch.normal(0,0.01, (n_head * d_head, d_model)), requires_grad=True)
        self.Wv3 = nn.Parameter(torch.normal(0,0.01, (n_head * d_head, d_model)), requires_grad=True)
        self.Wv4 = nn.Parameter(torch.normal(0,0.01, (n_head * d_head, d_model)), requires_grad=True)
        self.Wv5 = nn.Parameter(torch.normal(0,0.01, (n_head * d_head, d_model)), requires_grad=True)
        
        # self.Wv = nn.Parameter(torch.ones((n_head * d_head, d_model)), requires_grad=False)
        # self.Wv2 = nn.Parameter(torch.ones((n_head * d_head, d_model)), requires_grad=False)
        # self.bv = nn.Parameter(0.0 * torch.ones((n_head * d_head)), requires_grad=True)
        # nn.init.kaiming_uniform_(self.Wv, a=math.sqrt(5))
        self.nonlinact = nn.LeakyReLU(0.1)
        
        self.linear_layer = nn.Linear(d_model, n_head * d_head, bias=False)
        
        
        # self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        
        # self.layer_norm_v = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        h_val = h
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)
            
        c_val = c
        
        head_q = self.q_net(h) # Nt x bs x Nx
        head_k = self.k_net(c)
        # head_v = self.v_net(c)
        head_v = c.transpose(0,1)
        
        
        
        # True terms
        D_spacedu2dx2 = generate_Dspace(self.d_model, 1,1).to(device)
        D_spaceududx = generate_Dspace(self.d_model, 1,0).to(device)
        D_spaceu = generate_Dspace(self.d_model, 0,0).to(device)
        
        WDxdu2dx2 = Nx / ((2 * np.pi)) * D_spacedu2dx2 * self.Wv2
        WDududx = D_spaceududx * self.Wv
        WDu = D_spaceu
        
        head_v_du2dx2 = torch.matmul(head_v, WDxdu2dx2.T)
        head_v_ududx = torch.matmul(head_v, WDu.T) * torch.matmul(head_v, WDududx.T)
        
        # Sum all up
        head_v = head_v_du2dx2 + head_v_ududx
        
        
       
        head_v = head_v.transpose(0,1)
        
        # print(WDx[0:6,0:6])
        

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k)) # QK^T \in R^{N x N}
        attn_score.mul_(self.scale) # QK^T/sqrt(D)
        
#         if attn_mask is not None and attn_mask.any().item():
#             if attn_mask.dim() == 2:
#                 attn_score.masked_fill_(
#                     attn_mask[:,:,None,None], -float('inf'))
                
#                 # attn_score.masked_fill_(
#                 #     attn_mask2[:,:,None,None], 10000)
                
#                 # print(attn_score[:5,:5,0,0])
#             elif attn_mask.dim() == 3:
#                 attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

#         # [qlen x klen x bsz x n_head]
#         attn_score = F.softmax(attn_score, dim=1) # softmax (QK^T/sqrt(D)) = [-e^{qi^Tkj}/sum(e^{qi^Tkj})_j-]
        
#         if attn_mask is not None and attn_mask.any().item():
#             if attn_mask.dim() == 2:
#                 attn_score.masked_fill_(
#                     attn_mask[:,:,None,None], 0)
                
#                 # attn_score.masked_fill_(
#                 #     attn_mask2[:,:,None,None], 10000)
                
#                 print(attn_score[:5,:5,0,0])
#             elif attn_mask.dim() == 3:
#                 attn_score.masked_fill_(attn_mask[:,:,:,None], 0)
                
                
                
                
        attn_score =  soft_max_mask(ntokens, multi_step_num)[:,:,None,None] * attn_score

        # print(attn_prob[:5,:5,0,0])
        attn_prob = attn_score / (torch.sum(attn_score, dim=1, keepdim=True) + 1e-6)
        # attn_prob = attn_score  
#         print(attn_prob[:5,:5,0,0])
        
        # attn_prob = self.dropatt(attn_prob)
        
        
        
        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        output = attn_vec
        
        
        
        # output2 = torch.matmul(c_val, D_space) # This is satisfied the incremental step
        
        # print(output2 - output)
        
        # pdb.set_trace()
        # output = self.o_net(attn_vec)
        
        # output = self.drop(output)

        # if self.pre_lnorm:
        #     # residual connection
        #     output = h + attn_out
        # else:
        #     # residual connection + layer normalization
        #     output = self.layer_norm(h + attn_out)

        return output
    
class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(n_head=nhead, d_model=d_model, d_head=d_model, dropout=dropout, dropatt=0, 
                 pre_lnorm=norm_first)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, deltat: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x  = x.to(device) + (deltat[None,:,None]).to(device) * self._sa_block(x.to(device), src_mask.to(device), src_key_padding_mask) # This is to capture the time scheme
        
        
        
        
        
        return x


    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask)
        
        # return self.dropout1(x)
        
        return x


    
class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, deltat: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, deltat, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerModel(nn.Module):
    
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
#         import pdb; pdb.set_trace()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, norm_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, deltat: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        
        # src = self.pos_encoder(src) # shape [seq_len, batch_size, d_model] 
        
        output = self.transformer_encoder(src, deltat, src_mask) # shape [seq_len, batch_size, d_model]
        
        # output = self.decoder(output) # shape [seq_len, batch_size, ntokens]
        
        return output


# %%
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # import pdb; pdb.set_trace()
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# %%
def get_batch(source: Tensor, delta_t: Tensor, i: int, bs: int, jump_step: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    data    = source[0:-jump_step,i:i+bs,:]
    
    a,b,c = data.shape[0], data.shape[1], data.shape[2] 
    targets = torch.Tensor(a,b,c,jump_step)
    
    for iii in range(jump_step-1):
        targets[:,:,:,iii] = source[1+iii:-(jump_step - iii  - 1 ),i:i+bs,:]
        
    targets[:,:,:,jump_step - 1] = source[jump_step:,i:i+bs,:]
    dt_batch = delta_t[0,i:i+bs,0]

    return data, targets, dt_batch


# %%
batch_size = 1
batch_size_val = num_test
jump_step = 1
pred_steps = 101


ntokens = Nt - jump_step + 1 # size of vocabulary
emsize = Nx+1  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


# %%
import copy
import time

criterion = nn.MSELoss()

lr = .0001  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# lr = 1.0  # learning rate
# optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 45
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(ntokens).to(device)

    num_batches = train_data.size(1) // batch_size

    for batch, i in enumerate(range(0, train_data.size(1), batch_size)):
        
        data, targets, deltat = get_batch(train_data, train_data_dt, i, batch_size, jump_step)
        data, targets, deltat= data.to(device), targets.to(device), deltat.to(device)
        output = data
        
        loss = 0
        for iii in range(jump_step):
            output = model(output, deltat, src_mask)
            loss += criterion(output.to(device), targets[:,:,:,iii])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            # lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.6f}')
            total_loss = 0
            start_time = time.time()

            

def evaluate_seq(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(ntokens).to(device)
    
    with torch.no_grad():
        for i in range(0, eval_data.size(1), batch_size_val):
            data_whole, targets_jumps, deltat_test = get_batch(eval_data, test_data_dt, i, batch_size_val, jump_step)
            targets_whole = targets_jumps[:,:,:,0]
            
            data = torch.zeros(data_whole.shape)
            
            data_whole = data_whole.to(device)
            targets_whole = targets_whole.to(device)
           
            targets = torch.zeros(targets_whole.shape)
            output = torch.zeros((ntokens + pred_steps, Nx + 1))
            
            for step_time in range(0,ntokens + pred_steps):
                if step_time < ntokens:
                    if step_time == 0:
                        data[step_time,:,:] = data_whole[step_time,:,:]
                        output[step_time,:] = data[step_time,0,:]
                    if step_time != 0:
                        data[step_time,:,:] = data_next[step_time-1,:,:] 
                        output[step_time,:] = data[step_time,0,:]
                    
                    # data[step_time,:,:] = data_whole[step_time,:,:]
                    targets[step_time,:,:] = targets_whole[step_time,:,:]
                    data_next = model(data, deltat_test, src_mask)
                    total_loss = criterion(data.to(device), targets.to(device))
                    
                if step_time >= ntokens:
                    data = data_next
                    output[step_time,:] = data[-1,0,:]
                    data_next = model(data, deltat_test, src_mask)
                    
    return total_loss / (len(eval_data) - 1), output

# %%
best_val_loss = float('inf')
epochs = 1000
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss, _ = evaluate_seq(model, test_data_same)
    
#     val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.6f}')
    
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)




# %%
test_loss, solution_same_predict = evaluate_seq(best_model, test_data_same)
solution_same_predict.shape


# %%
plt.figure(figsize=(10,6))
sample_cases = 0
Step_outs = [0, 150]
color = ['r', 'b', 'g', 'm', 'k', 'c']

for step in range(len(Step_outs)):
    plt.plot(x, solution_same_predict[Step_outs[step],:].cpu(), '--o' + color[step], dashes=(3, 10), markevery= 50, label = 'Pred Step_' + str(Step_outs[step]))
    plt.plot(x, test_data_same[Step_outs[step], sample_cases, :],'--x' + color[step], dashes=(3, 6), markevery= 30,label = 'True Step_' + str(Step_outs[step]))

for step in [250]:
    plt.plot(x, solution_same_predict[step,:].cpu(), '-o', dashes=(3, 10), markevery= 50, label = 'Out_Pred Step_' + str(step))

plt.legend(loc = 'lower left', ncol = 2)
plt.title('Euler two terms')

file_name = 'Two_Euler_Nx_' + str(Nx) + '_Nt_'+ str(Nt) + 'Out_pred' + str(pred_steps) +'in_range.png'
plt.savefig('figs/' + file_name)
Solution_samples_array= pd.DataFrame({'samples' : (solution_same_predict.cpu()).flatten()})
Solution_samples_array.to_csv('Result_data_file/' + file_name + '.csv', index=False)











