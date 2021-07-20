import torch.nn as nn
from utils import setparams
from PWLA import PWLA2d
'''Testing the update function, needed after the first phase'''

m=nn.Sequential(
    nn.Conv2d(10,10,1),
    PWLA2d(),
    nn.Conv2d(10,10,1),
    PWLA2d(),
    nn.Conv2d(10,10,1),
    PWLA2d(),
)
for step,i in enumerate(m.children()):
    try:
        print(i.Br)
        i=setparams(i)
    except:pass
for step,i in enumerate(m.children()):
    try:
        print(i.Br)
        i=setparams(i)
    except:pass
