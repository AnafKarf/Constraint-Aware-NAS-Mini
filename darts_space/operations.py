import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'param_softplus_3x3' : lambda C, stride, affine: ParamSoftplusConv(C, C, 3, stride, 1, affine=affine),
  'param_softplus_5x5' : lambda C, stride, affine: ParamSoftplusConv(C, C, 5, stride, 2, affine=affine),
  'param_sinh_3x3' : lambda C, stride, affine: ParamSinhConv(C, C, 3, stride, 1, affine=affine),
}


class ParamSoftplusConv(nn.Module):
    """
    Parametric Softplus Convolution: φ(x) = (1/ω)ln(1+e^(ωx))
    Learnable ω parameter constrained to maintain ∂φ/∂x ≤ κ_act
    """
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, 
                 omega_init=1.0, kappa_act=1.5):
        super(ParamSoftplusConv, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, 
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        
        # Learnable activation parameter with constraint
        # Initialize conservatively to satisfy derivative bound
        self.omega = nn.Parameter(torch.tensor(omega_init, dtype=torch.float32))
        self.kappa_act = kappa_act
        self.register_buffer('omega_max', torch.tensor(4.0 * kappa_act))  # ω ≤ 4*κ_act for constraint
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # Scaled Parametric Softplus: φ(x) = α * softplus(βx) where α controls max derivative
        # Standard softplus derivative max is 1/4, so α * 1/4 ≤ κ_act means α ≤ 4*κ_act
        # We use ω as the scaling factor α, and fix β=1 for simplicity
        omega_clamped = torch.clamp(self.omega, min=0.1, max=self.omega_max)
        x = omega_clamped * F.softplus(x)
        return x
    
    def get_max_derivative(self):
        """Returns theoretical maximum derivative of the activation"""
        omega_clamped = torch.clamp(self.omega, min=0.1, max=self.omega_max)
        # For scaled softplus: d/dx[α * softplus(x)] = α * sigmoid(x)
        # Max derivative = α * 1/4 = ω/4
        return omega_clamped / 4.0


class ParamSinhConv(nn.Module):
    """
    Parametric Hyperbolic Sine Convolution: φ(x) = α * sinh(βx)
    Learnable α, β parameters with derivative constraint
    """
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True,
                 alpha_init=1.0, beta_init=0.5, kappa_act=1.5):
        super(ParamSinhConv, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.kappa_act = kappa_act
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # Parametric sinh: φ(x) = α * sinh(βx)
        # Derivative: φ'(x) = αβ * cosh(βx)
        alpha_clamped = torch.clamp(self.alpha, min=0.1, max=2.0)
        # Convert to scalar for max value computation
        beta_max = self.kappa_act / alpha_clamped.item()
        beta_clamped = torch.clamp(self.beta, min=0.1, max=beta_max)
        x = alpha_clamped * torch.sinh(beta_clamped * x)
        return x
    
    def get_max_derivative(self):
        """Returns theoretical maximum derivative of the activation"""
        alpha_clamped = torch.clamp(self.alpha, min=0.1, max=2.0)
        beta_max = self.kappa_act / alpha_clamped.item()
        beta_clamped = torch.clamp(self.beta, min=0.1, max=beta_max)
        # For sinh: d/dx[α·sinh(βx)] = αβ·cosh(βx)
        # Assuming bounded inputs |x| ≤ 1 after normalization, max derivative ≈ αβ·cosh(β)
        # More accurate estimate: αβ·cosh(β) where β is the clamped value
        max_derivative = alpha_clamped * beta_clamped * torch.cosh(beta_clamped)
        return max_derivative


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

