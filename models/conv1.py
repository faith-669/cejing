from turtle import forward
import torch.nn as nn

lable_num = 10

class CNNNet(nn.Module):    
    def __init__(self): 
        super(CNNNet, self).__init__()  

        self.conv_layer = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

        )

        self.fc = nn.Sequential(
            nn.Linear(79872,256),
            nn.ReLU(inplace = True),
            nn.Dropout(),

            nn.Linear(256,256),
            nn.ReLU(inplace = True),
            nn.Dropout(),

            nn.Linear(256, lable_num)

        )       
    
    def forward(self,x):
        x = self.conv_layer(x)
        #print(f"特征图大小：{x.shape}")
        x = x.view(-1, 79872)
        x = self.fc(x)
        return x
