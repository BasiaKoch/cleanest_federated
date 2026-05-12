import torch
from torch import nn
import torch.nn.functional as F


class LogitAdjustedCrossEntropy(nn.Module):
    """Training-time logit adjustment.

    Formula: CE(z - tau * log(pi), y), where pi is the class prior.
    Since log(pi) is more negative for minority classes, subtracting it raises
    minority logits more than majority logits and reduces majority-class bias.
    """

    def __init__(self, class_priors, tau: float = 1.0):
        super().__init__()
        class_priors = torch.as_tensor(class_priors, dtype=torch.float32)
        assert torch.all(class_priors > 0), 'All priors must be positive'
        class_priors = class_priors / class_priors.sum()
        self.register_buffer('adjustment', tau * torch.log(class_priors + 1e-12))

    def forward(self, logits, targets):
        return F.cross_entropy(logits - self.adjustment.to(logits.device), targets.view(-1).long())


def post_hoc_logit_adjustment(logits, class_priors, tau: float = 1.0):
    class_priors = torch.as_tensor(class_priors, dtype=logits.dtype, device=logits.device)
    class_priors = class_priors / class_priors.sum()
    return logits - tau * torch.log(class_priors + 1e-12)


if __name__ == '__main__':
    logits = torch.tensor([[5.0, 1.0, 1.0]])
    priors = torch.tensor([0.8, 0.1, 0.1])
    adj = post_hoc_logit_adjustment(logits, priors, tau=1.0)
    # class 0: 5 - log(0.8) = 5.223; class 1: 1 - log(0.1) = 3.303
    assert adj[0, 1] - logits[0, 1] > adj[0, 0] - logits[0, 0]
    assert torch.allclose(post_hoc_logit_adjustment(logits, priors, tau=0.0), logits)
    uniform = torch.tensor([1/3, 1/3, 1/3])
    adj_uniform = post_hoc_logit_adjustment(logits, uniform)
    assert torch.argmax(adj_uniform, dim=1).item() == torch.argmax(logits, dim=1).item()
    print('Logit adjustment tests passed')
