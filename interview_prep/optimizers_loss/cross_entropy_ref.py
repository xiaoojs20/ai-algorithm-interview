import torch
import torch.nn as nn

class StableCrossEntropy(nn.Module):
    """
    Cross Entropy with Log-Sum-Exp Trick
    
    LaTeX Formula:
    $ \log \sum e^{x_j} = A + \log \sum e^{x_j - A} $
    """
    def forward(self, logits, targets):
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1, keepdim=True))
        target_logits = torch.gather(logits, -1, targets.unsqueeze(-1))
        return (-target_logits + log_sum_exp).mean()
