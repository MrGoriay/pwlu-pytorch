import torch, torch.nn as nn

def setparams(act):
    act.Bl = torch.nn.Parameter(act.running_mean.detach() - 3*act.running_var.detach())
    act.Br = torch.nn.Parameter(act.running_mean.detach() + 3*act.running_var.detach())
    act.Yidx = torch.nn.Parameter(nn.functional.relu(torch.linspace(act.Bl.item(),act.Br.item(),act.N+1)))
    return act