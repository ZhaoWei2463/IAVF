import torch
import torch.nn as nn
import torch.nn.functional as F

class MIM(nn.Module):
    def __init__(self,C,dropout):
        super(MIM,self).__init__()
        #down_up
        self.conv1 = nn.Conv2d(C, 2*C, kernel_size=3, stride=2, padding=1)      #(C,H,W) to (2C,H/2,W/2)
        self.bn1 = nn.BatchNorm2d(2*C)
        self.deconv1 = nn.ConvTranspose2d(2*C, C, kernel_size=4, stride=2, padding=1)   #(2C,H/2,W/2) to (C,H,W)
        self.bn_d1 = nn.BatchNorm2d(C)        
        self.conv_a = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0)
        #up_down
        self.deconv2 = nn.ConvTranspose2d(C, 2*C, kernel_size=4, stride=2, padding=1)   #(C,H,W) to (2C,2H,2W)
        self.bn_d2 = nn.BatchNorm2d(2*C)
        self.conv2 = nn.Conv2d(2*C, C, kernel_size=3, stride=2, padding=1)    #(2C,2H,2W) to (C,H,W)
        self.bn2 = nn.BatchNorm2d(C)
        self.conv_b = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0)

        self.conv3 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)        #(C,H,W) to (C,H,W)
        self.bn3 = nn.BatchNorm2d(C)
        self.conv_c = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(3*C, C, kernel_size=3,stride=1, padding=1)       #(3C,H,W) to (C,H,W)
        self.bn4 = nn.BatchNorm2d(C)
        
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x):            #(batchsize,c,N,F)
        z1_en = F.relu(self.bn1(self.conv1(x)))
        z1_de = F.relu(self.bn_d1(self.deconv1(z1_en)))
        z1 = z1_de + x 
        z1 = self.conv_a(z1)
        z1 = self.dropout(z1)

        z2_de = F.relu(self.bn_d2(self.deconv2(x)))
        z2_en = F.relu(self.bn2(self.conv2(z2_de)))
        z2 = z2_en + x 
        z2 = self.conv_b(z2)
        z2 = self.dropout(z2)

        z3 = F.relu(self.bn3(self.conv3(x)))
        z3 = z3 + x
        z3 = self.conv_c(z3)
        z3 = self.dropout(z3)

        Z = torch.cat((z1,z2,z3),dim=1)
        return F.relu(self.bn4(self.conv4(Z)))                     #(batchsize,c,N,F)
    

# x = torch.rand(64,3,32,32)
# c = MIM(3,0.2)
# z = c(x)
# print(z.size())