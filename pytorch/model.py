import torch.nn as nn
import torch

class QHD_Model(nn.Module):
    def __init__(self, seperate) -> None:
        super(QHD_Model, self).__init__()
    
        self.mode = seperate 
        
        if not seperate:
            self.kernels = nn.Sequential(*[
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 6),
                nn.Sigmoid()
            ])
        
        else:
            self.kernels = []
            self.kernels.extend(
                [nn.Sequential(*[
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 6),
                nn.Sigmoid()]) for i in range(6)]
            )
    
    def forward(self, x):
        if not self.mode:
            out = self.kernels(x)
        else:
            out = self.kernels[0](x)
            for i in range(1, len(self.kernels)):
                out = torch.concat((out, self.kernels[i](x)), axis = 1)
        
        return out

# x = torch.randn((3,768))
# model = QHD_Model(True)
# print(model(x).shape)