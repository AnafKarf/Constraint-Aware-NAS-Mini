import torch
import torch.nn as nn
from darts_space.operations import ParamSoftplusConv, ParamSinhConv

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
    
    
def adjust_learning_rate(args, optimizer, epoch, learning_rate, epochs=None, dataset=None, lr_scheduler=None):
    
    if dataset == 'ImageNet16-120':
        if args.lr_scheduler == 'cosine':
            assert lr_scheduler is not None
            lr_scheduler.step()
        elif args.lr_scheduler == 'linear':
            if epochs-epoch > 5:
                lr = learning_rate * (epochs - 5 - epoch) / (epochs - 5)
            else:
                lr = learning_rate * (epochs - epoch) / ((epochs - 5) * 5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            NotImplementedError()
            
        if epoch < 5 and args.batch_size > 256:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate * (epoch + 1) / 5.0
    else:
        lr = learning_rate
        if epoch >= 99:
            lr = learning_rate * 0.1
        if epoch >= 149:
            lr = learning_rate * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def spectral_norm_clip(module, beta=1.0, eps=1e-12):
    """
    Clip spectral norm of weight matrices to satisfy σ₁(W) ≤ β
    Uses exact SVD for accuracy during training (can be optimized later)
    """
    if not hasattr(module, 'weight') or module.weight is None:
        return 0.0
    
    W = module.weight.data
    if W.dim() < 2:
        return 0.0
    
    # Reshape to 2D matrix
    W_reshaped = W.view(W.size(0), -1)
    
    with torch.no_grad():
        # Use exact SVD for accurate spectral norm computation
        try:
            u, s, v = torch.svd(W_reshaped)
            sigma = s[0].item()
            
            # Clip if necessary
            if sigma > beta:
                # Scale down the weight matrix
                scale_factor = beta / sigma
                W_reshaped.mul_(scale_factor)
                # Copy back to original shape
                module.weight.data.copy_(W_reshaped.view_as(W))
                return sigma
            
            return sigma
            
        except Exception as e:
            # Fallback to power iteration if SVD fails
            u = torch.randn(W_reshaped.size(0), device=W.device, dtype=W.dtype)
            u = u / (torch.norm(u) + eps)
            
            for _ in range(3):  # More iterations for better approximation
                v = torch.matmul(W_reshaped.t(), u)
                v = v / (torch.norm(v) + eps)
                u = torch.matmul(W_reshaped, v)
            
            sigma = torch.norm(u)
            
            if sigma > beta:
                W_reshaped.div_(sigma / beta)
                module.weight.data.copy_(W_reshaped.view_as(W))
                return sigma.item()
            
            return sigma.item()


def project_activation_constraints(model, kappa_act=1.5):
    """
    Project learnable activation parameters to satisfy ∂φ/∂x ≤ κ_act
    """
    constraint_violations = []
    
    for name, module in model.named_modules():
        if isinstance(module, ParamSoftplusConv):
            # For scaled softplus: φ(x) = ω * softplus(x), max derivative = ω/4
            # To satisfy ω/4 ≤ κ_act, we need ω ≤ 4*κ_act
            omega_max = 4.0 * kappa_act
            if module.omega.data > omega_max:
                constraint_violations.append((name, 'softplus', module.omega.data.item()))
                module.omega.data.clamp_(max=omega_max)
                
        elif isinstance(module, ParamSinhConv):
            # For sinh: max derivative is αβ*cosh(βx), bounded by αβ*exp(β*|x_max|)
            # Conservative bound: αβ ≤ κ_act (assuming |x| ≤ 1 after normalization)
            alpha_val = module.alpha.data
            beta_val = module.beta.data
            if alpha_val * beta_val > kappa_act:
                # Scale both parameters proportionally
                scale_factor = kappa_act / (alpha_val * beta_val)
                module.alpha.data.mul_(torch.sqrt(scale_factor))
                module.beta.data.mul_(torch.sqrt(scale_factor))
                constraint_violations.append((name, 'sinh', (alpha_val * beta_val).item()))
    
    return constraint_violations


def apply_constraints(model, beta=1.0, kappa_act=1.5, log_violations=False):
    """
    Apply all constraints: spectral norm clipping and activation parameter projection
    
    Args:
        model: PyTorch model
        beta: Maximum spectral norm bound
        kappa_act: Maximum activation derivative bound
        log_violations: Whether to return constraint violation info
    
    Returns:
        Dictionary with constraint violation statistics if log_violations=True
    """
    spectral_violations = []
    activation_violations = []
    
    # Apply spectral norm constraints to all conv/linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            sigma = spectral_norm_clip(module, beta)
            if sigma > beta:
                spectral_violations.append((name, sigma))
    
    # Apply activation parameter constraints
    activation_violations = project_activation_constraints(model, kappa_act)
    
    if log_violations:
        return {
            'spectral_violations': spectral_violations,
            'activation_violations': activation_violations,
            'max_spectral_norm': max([s[1] for s in spectral_violations] + [0.0]),
            'num_spectral_violations': len(spectral_violations),
            'num_activation_violations': len(activation_violations)
        }
    
    return None


def get_constraint_stats(model):
    """
    Get current constraint statistics without modifying the model
    """
    stats = {
        'spectral_norms': [],
        'activation_params': [],
        'max_spectral_norm': 0.0,
        'activation_derivatives': []
    }
    
    # Collect spectral norms
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
            W = module.weight.data
            if W.dim() >= 2:
                W_reshaped = W.view(W.size(0), -1)
                sigma = torch.svd(W_reshaped)[1][0].item()  # Largest singular value
                stats['spectral_norms'].append((name, sigma))
                stats['max_spectral_norm'] = max(stats['max_spectral_norm'], sigma)
    
    # Collect activation parameters
    for name, module in model.named_modules():
        if isinstance(module, ParamSoftplusConv):
            omega_val = module.omega.data.item()
            max_deriv = omega_val / 4.0  # Theoretical max derivative
            stats['activation_params'].append((name, 'softplus', omega_val))
            stats['activation_derivatives'].append((name, 'softplus', max_deriv))
            
        elif isinstance(module, ParamSinhConv):
            alpha_val = module.alpha.data.item()
            beta_val = module.beta.data.item()
            # Conservative estimate of max derivative
            max_deriv = alpha_val * beta_val  # Assuming normalized inputs
            stats['activation_params'].append((name, 'sinh', alpha_val, beta_val))
            stats['activation_derivatives'].append((name, 'sinh', max_deriv))
    
    return stats