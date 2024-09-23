import torch.nn as nn
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union
import math
import torch
from training.nn import Linear, GroupNorm


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class LoraInjectedConv(nn.Module):

    def __init__(
            self, dim, in_channels, out_channels, bias, kernel_size, padding, lora_rank, num_basis, mlp_width,
            emb_channels, scale=1.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.lora_rank = lora_rank

        self.conv = conv_nd(dim, in_channels, out_channels, kernel_size, padding=padding, bias=bias)

        self.lora_down = conv_nd(dim, in_channels, lora_rank * num_basis, kernel_size, padding=padding, bias=False)
        self.lora_up = conv_nd(dim, lora_rank * num_basis, out_channels, kernel_size, padding=padding, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / lora_rank)
        nn.init.zeros_(self.lora_up.weight)

        # If we choose to use bias
        self.bias = nn.Parameter(torch.zeros((lora_rank * num_basis, out_channels)))


        self.scale = scale

        # If we choose to use the trainable scale
        # self.scale = nn.Parameter(torch.randn(1))

        self.comp_weights = [nn.Sequential(
            Linear(in_features=emb_channels,
                   out_features=mlp_width[0]),
            GroupNorm(num_channels=mlp_width[0]),
            nn.SiLU()
        )]

        for i in range(1, len(mlp_width)):
            self.comp_weights.append(
                nn.Sequential(
                    Linear(in_features=mlp_width[i - 1],
                           out_features=mlp_width[i]),
                    GroupNorm(num_channels=mlp_width[i]),
                    nn.SiLU(),
                )
            )

        self.comp_weights.append(Linear(in_features=mlp_width[-1], out_features=num_basis))

        self.comp_weights = nn.ModuleList(self.comp_weights)

        self.bypass = False


    def set_bypass(self, bypass):
        self.bypass = bypass
        return

    def forward(self, input, emb):
        if self.bypass:
            return self.conv(input)

        comp_weights = self.comp_weights[0](emb)
        for i in range(1, len(self.comp_weights)):
            comp_weights = self.comp_weights[i](comp_weights)

        if self.dim == 1:
            comp_weights = torch.repeat_interleave(comp_weights, self.lora_rank, dim=1).unsqueeze(-1)
        elif self.dim == 2:
            comp_weights = torch.repeat_interleave(comp_weights, self.lora_rank, dim=1).unsqueeze(-1).unsqueeze(-1)

        conv_out = self.conv(input)
        lora_out = self.lora_up(comp_weights * self.lora_down(input))


        # If we choose to use bias
        if self.dim == 1:
            lora_out += torch.matmul(comp_weights.squeeze(), self.bias).unsqueeze(-1)
        elif self.dim == 2:
            lora_out += torch.matmul(comp_weights.squeeze(), self.bias).unsqueeze(-1).unsqueeze(-1)

        lora_scale = self.scale

        # If we choose to use tanh
        # lora_scale = nn.Tanh()(self.scale)

        return conv_out + lora_scale * lora_out


DEFAULT_TARGET_REPLACE = {"UNetBlock"}


def _find_modules(
        model,
        ancestor_class: Optional[Set[str]] = None,
        search_class: List[Type[nn.Module]] = [nn.Conv1d, nn.Conv2d],
        exclude_children_of: Optional[List[Type[nn.Module]]] = [
            LoraInjectedConv,
        ],
):
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                        [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def set_bypass_for_lora(
        model: nn.Module,
        bypass: bool,
        target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
        names = ["qkv", "proj"],
        verbose: bool = True,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    for _module, name, _child_module in _find_modules(
            model, target_replace_module, search_class=[LoraInjectedConv], exclude_children_of=[]
    ):
        if name in names:
            _child_module.set_bypass(bypass)
            if verbose:
                print(f"Setting bypass as {bypass} for {name}")
    return


def inject_trainable_lora(
        model: nn.Module,
        target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
        _names = ["qkv", "proj"],
        lora_rank: int = 4,
        num_basis: int = 18,
        mlp_width: List = [128, 64],
        loras=None,  # path to lora .pt
        verbose: bool = False,
        scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    from .networks import Conv2d
    for _module, name, _child_module in _find_modules(
            model, target_replace_module, search_class=[Conv2d]
    ):
        if name in _names:

            weight = _child_module.weight
            bias = _child_module.bias
            if verbose:
                print("LoRA Injection : injecting lora into ", name)
                print("LoRA Injection : weight shape", weight.shape)

            _tmp = LoraInjectedConv(
                2,
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.bias is not None,
                _child_module.weight.shape[-1],
                _child_module.weight.shape[-1] // 2,
                lora_rank,
                num_basis,
                mlp_width,
                _module.emb_channels,
                scale
            )
            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            require_grad_params.append(_module._modules[name].lora_up.parameters())
            require_grad_params.append(_module._modules[name].lora_down.parameters())
            require_grad_params.append(_module._modules[name].comp_weights.parameters())

            if loras != None:
                _module._modules[name].t_lora_up.weight = loras.pop(0)
                _module._modules[name].t_lora_down.weight = loras.pop(0)


            names.append(name)

        for params in require_grad_params:
            for p in params:
                p.requires_grad = True

    return require_grad_params, names

