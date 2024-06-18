import torch
from torch import nn
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True):
        super().__init__()
        # Linear layer
        self.lin = nn.Linear(in_features=in_features, out_features=out_features)
        # Weight init
        nn.init.normal_(self.lin.weight, std=0.02)
        # Batch normalization if applicable
        self.bn = nn.BatchNorm1d(out_features) if batchnorm else nn.Identity()

    def forward(self, x):
        x = self.lin(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class NeuralNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Linear layers
        self.l1 = LinearLayer(in_features=input_shape, out_features=4000, batchnorm=False)
        self.l2 = LinearLayer(in_features=4000, out_features=2000, batchnorm=False)
        self.l3 = LinearLayer(in_features=2000, out_features=1000, batchnorm=False)
        self.l4 = LinearLayer(in_features=1000, out_features=500, batchnorm=False)
        self.l5 = LinearLayer(in_features=500, out_features=250, batchnorm=False)
        self.out = nn.Linear(in_features=250, out_features=10)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        # Verificar las dimensiones de las entradas y salidas de cada capa
        print(f"Input: {x.shape}")
        l1_out = self.l1(x)
        print(f"l1_out: {l1_out.shape}")
        l2_out = self.l2(l1_out)
        print(f"l2_out: {l2_out.shape}")
        l3_out = self.l3(l2_out)
        print(f"l3_out: {l3_out.shape}")
        l4_out = self.l4(l3_out)
        print(f"l4_out: {l4_out.shape}")
        l5_out = self.l5(l4_out)
        print(f"l5_out: {l5_out.shape}")
        # Output layer
        out = self.out(l5_out)
        print(f"Output: {out.shape}")
        return out
        
class NeuralNet_mfcc(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Linear layers
        self.l1 = LinearLayer(in_features=input_shape*16, out_features=20, batchnorm=False)
        self.l2 = LinearLayer(in_features=20, out_features=15, batchnorm=False)
        self.l3 = LinearLayer(in_features=15, out_features=15, batchnorm=False)
        self.l4 = LinearLayer(in_features=15, out_features=12, batchnorm=False)
        self.l5 = LinearLayer(in_features=12, out_features=12, batchnorm=False)
        self.out = nn.Linear(in_features=12, out_features=10)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        print(f"Input: {x.shape}")
        l1_out = self.l1(x)
        print(f"l1_out: {l1_out.shape}")
        l2_out = self.l2(l1_out)
        print(f"l2_out: {l2_out.shape}")
        l3_out = self.l3(l2_out)
        print(f"l3_out: {l3_out.shape}")
        l4_out = self.l4(l3_out)
        print(f"l4_out: {l4_out.shape}")
        l5_out = self.l5(l4_out)
        print(f"l5_out: {l5_out.shape}")
        # Output layer
        out = self.out(l5_out)
        print(f"Output: {out.shape}")
        return out


