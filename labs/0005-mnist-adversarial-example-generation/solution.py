import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example for input x.
    
    Args:
        model: Pre-trained classifier (already in eval mode)
        x: Input image tensor, shape (1, 1, 28, 28), values in [0,1]
        y: True label, shape (1,) or scalar
        epsilon: L∞ perturbation budget
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
    
    Returns:
        x_adv: Adversarial example, same shape as x, satisfying:
               - ||x_adv - x||_∞ ≤ epsilon
               - x_adv values in [0, 1]
               - model(x_adv).argmax() != y (ideally)
    '''
    # TODO: Implement your adversarial attack here
    # Hint: Use gradients of loss w.r.t. input x
    for param in model.parameters():
        param.grad = None
    x.requires_grad_(True)
    y_pred = model(x)
    top_answers = torch.topk(y_pred,2)

    #if (isinstance(y, float) and top_answers.indices[0] != y) or (top_answers.indices[0] != y.item()) :
    #    return x
    loss = criterion(y_pred, y)
    loss.backward()
    change = torch.sign(x.grad) * epsilon
    return torch.clamp(x + change,0,1)
