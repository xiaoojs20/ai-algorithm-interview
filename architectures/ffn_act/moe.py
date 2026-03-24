import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    """
    Practice Template: Mixture of Experts (MoE)
    
    Complete the following implementation:
    1. Gating Network (Router)
    2. Top-K Selection
    3. Expert Dispatching (Efficiently)
    4. Optional: Load Balancing Loss
    """
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # TODO: Define Router
        # self.router = ...
        
        # TODO: Define Experts (ModuleList)
        # self.experts = ...

    def forward(self, x):
        """
        Input: (batch, seq, d_model)
        Output: (batch, seq, d_model), aux_loss
        """
        # TODO: Implement MoE logic
        pass

if __name__ == "__main__":
    # Test your implementation
    pass
