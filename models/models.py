import torch.nn as nn
import torch.nn.functional as F


class ffwReg(nn.Module):
    
    def __init__(self, n_inp, n_out, n_hidd, n_hidd_layers,
                 act_fun, dropout=False):
        
        super(ffwReg, self).__init__()
        self.fcInput = nn.Linear(n_inp, n_hidd)
        self.linears = nn.ModuleList([nn.Linear(n_hidd, n_hidd) for n in range(n_hidd_layers)])
        self.fcOutput= nn.Linear(n_hidd, n_out)
        self.act_fun = act_fun

        self.dropout = dropout
        if self.dropout:
            self.dropLayer = nn.Dropout(p=0.5)

    def forward(self, x):
        
        if self.act_fun == 'relu':
            act = F.relu
        if self.act_fun == 'sigmoid':
            act = F.sigmoid
        if self.act_fun == 'tanh':
            act = F.tanh
            
        if self.dropout:
            x = self.dropLayer(act(self.fcInput(x)))
            for l in self.linears:
                x = self.dropLayer(act(l(x)))
        else:
            x = act(self.fcInput(x))
            for l in self.linears:
                x = act(l(x))

        x = self.fcOutput(x)
                
        return x