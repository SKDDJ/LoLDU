import math
from functools import partial
from typing import Any, Optional, Union
import torch
from torch import svd_lowrank
import torch.nn.utils.parametrize as parametrize
from torch import nn
import lightning as pl

import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


def ldu_decomposition(A):
    # Perform LU decomposition
    P, L, U = torch.linalg.lu(A)
    # Extract the diagonal elements of U to form D
    D = torch.diagonal(U)
    # Normalize U to get the unit diagonal upper triangular matrix
    U = U / D.unsqueeze(-1)
    return P, L, D, U


class LoLDUParametrization(pl.LightningModule):
    def __init__(
        self,
        fan_in,
        fan_out,
        fan_in_fan_out=False,
        rank=4,
        lora_dropout_p=0.0,
        lora_alpha=8,
        layer_module=None,
        init_loldu_weights="ldu",
        init_method="lu",
    ):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        # 假设layer_module是一个已存在的层，我们用它的权重尺寸初始化layer_weights
        layer_weights = layer_module.weight

        self.layer_weights = layer_weights
        del layer_weights

        self.layer_weights.data = layer_module.weight.data

        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)

        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.init_method = init_method
        self.lora_dropout_p = lora_dropout_p

        # 注册一个前向传播前的钩子，以自动更新权重一次
        self._forward_pre_hook_handle = self.register_forward_pre_hook(
            self._update_weights_once
        )
        # 添加一个额外的属性来标记是否更新过权重
        self._updated = False
        # lu init
        self.init_loldu_weights = init_loldu_weights

        # todo: test removing the code below
        self.forward_fn = self.loldu_forward

    def _update_weights_once(self, *args):
        if not self._updated:  # 只有在未更新权重的情况下才执行
            # initialization of A and B
            # note: because we may use weight tying, so we have to define the lora_X as nn.Parameter not the nn.Linear
            self.lora_A = nn.Parameter(
                torch.zeros(self.swap((self.rank, self.fan_in)))
            )  # U
            self.lora_B = nn.Parameter(
                torch.zeros(self.swap((self.fan_out, self.rank)))
            )  # L
            if self.init_loldu_weights == "ldu":
                self.P = nn.Parameter(
                    torch.zeros(self.swap((self.fan_out, self.fan_out)))
                )  # P
                self.vector_z = nn.Parameter(torch.ones(self.rank))
            # TODO: TEST init a nn.Parameter name scaling_factor that is a scalar
            self.scaling_factor = nn.Parameter(torch.tensor(self.scaling))

            self.get_residual_matrix()
            self._forward_pre_hook_handle.remove()  # 移除钩子
            self._updated = True  # 更新标记，防止再次执
        # 权重更新后立即移除钩子，确保只执行一次

    # @torch.no_grad()
    def get_residual_matrix(self):
        r = self.rank
        init_loldu_weights = self.init_loldu_weights

        weight = self.layer_weights
        dtype = weight.dtype
        # print("weight dtype", dtype)

        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        # print("new weight dtype", weight.dtype)

        # todo check if it is need to del teh U V S
        if init_loldu_weights == "ldu":
            P, L, D, U = ldu_decomposition(weight.data)
            # Extract the top r components

            Lr = L[:, :r]
            Dr = D[:r]
            Ur = U[:r]

            lora_A = Ur
            lora_B = Lr
            vector_z = Dr
            self.lora_A.data = lora_A
            self.lora_B.data = lora_B
            self.P.data = P
        elif init_loldu_weights == "lora":
            lora_A = torch.randn(self.swap((self.rank, self.fan_in)), dtype=dtype)
            lora_B = torch.zeros(self.swap((self.fan_out, self.rank)), dtype=dtype)

            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

            self.lora_A.data = lora_A
            self.lora_B.data = lora_B

        # only train vector_z
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        if init_loldu_weights == "ldu":
            self.P.requires_grad = False

        init_method = (
            self.init_method
        )  # 可选项: 'lu', 'kaiming_normal', 'kaiming_uniform', 'uniform', 'normal', 'constant', 'ones', 'zeros'
        if init_method == "lu":
            # print(vector_z)
            # print(vector_z.shape)
            # print(vector_z.item())
            self.vector_z.data = vector_z
        # elif init_method == 'kaiming_normal':
        #     nn.init.kaiming_normal_(self.vector_z, mode='fan_in', nonlinearity='relu')
        # elif init_method == 'kaiming_uniform':
        #     nn.init.kaiming_uniform_(self.vector_z, a=math.sqrt(5))
        elif init_method == "uniform":
            self.vector_z.data = vector_z
            mean_value = self.vector_z.mean().item()
            print(f"self.vector_z mean is {mean_value}")
            print(f"self.vector_z is {self.vector_z}")
            # write a if logic here, if the mean value is greater than 0 then ... otherwise ...
            if mean_value > 0:
                nn.init.uniform_(self.vector_z, a=-mean_value / 2, b=mean_value / 2)
            else:
                nn.init.uniform_(self.vector_z, a=mean_value / 2, b=-mean_value / 2)
            # nn.init.uniform_(self.vector_z, a=-mean_value/2, b=mean_value/2)
            # nn.init.uniform_(self.vector_z, a=-1, b=1)
        elif init_method == "s_uniform":
            nn.init.uniform_(self.vector_z, a=-1, b=1)
        elif init_method == "normal":
            self.vector_z.data = vector_z
            mean_value = self.vector_z.mean().item()
            std_value = self.vector_z.std().item()
            print(f"self.vector_z mean is {mean_value}")
            print(f"self.vector_z std is {std_value}")
            nn.init.normal_(self.vector_z, mean=mean_value, std=std_value)
        elif init_method == "s_normal":
            nn.init.normal_(self.vector_z, mean=0.0, std=1.0)
        elif init_method == "constant":
            self.vector_z.data = vector_z
            mean_value = self.vector_z.mean().item()
            print(f"self.vector_z mean is {mean_value}")
            nn.init.constant_(self.vector_z, mean_value)
            # nn.init.constant_(self.vector_z, 3.14)
        elif init_method == "ones":
            nn.init.ones_(self.vector_z)
        elif init_method == "zeros":
            nn.init.zeros_(self.vector_z)
        elif init_method == "lora":
            pass
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # drop out which won't be used
        self.lora_dropout = (
            nn.Dropout(p=self.lora_dropout_p)
            if self.lora_dropout_p > 0
            else lambda x: x
        )
        self.dropout_fn = self._dropout if self.lora_dropout_p > 0 else lambda x: x
        self.register_buffer(
            "lora_dropout_mask",
            torch.ones(self.swap((1, self.fan_in)), dtype=self.lora_A.dtype),
        )
        # keep the resmat even zeros because finally we need merge the resmat with the LDU but we can don't use the resmat in the forward
        if init_loldu_weights == "ldu":
            resmat = (
                self.layer_weights
                - self.scaling_factor * P @ lora_B @ torch.diag(vector_z) @ lora_A
            )
            resmat = resmat.to(dtype)
            self.layer_weights.data = resmat
            del resmat, lora_A, lora_B, vector_z, weight

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def loldu_forward(self, X):
        torch_X_dtype = X.dtype  # 16
        # print("I am forwarding"*20)
        # print(f"self.P device is {self.P.device}")
        # print(f"L(loraB) device is {self.lora_B.device}")
        # print(f"D device is {self.vector_z.device}")
        # print(f"U(loraA) device is {self.lora_A.device}")
        if self.init_loldu_weights == "ldu":
            diag_z = torch.diag(self.vector_z)  # 32
            # print("self.scaling_factor init", self.scaling_factor)
            result = self.scaling_factor * self.P @ self.lora_B @ diag_z @ self.lora_A
        elif self.init_loldu_weights == "lora":
            result = self.scaling_factor * self.lora_B @ self.lora_A
        else:
            raise ValueError(
                f"Unknown initialization method: {self.init_loldu_weights}"
            )

        result = result.to(torch_X_dtype)  # 32 -> 16
        if self.init_loldu_weights == "ldu":
            del diag_z
        # omit the original weights only return the LDU matrix
        return X + result

    def forward(self, X):
        return self.forward_fn(X)

    @classmethod
    def from_linear(
        cls,
        layer,
        rank=4,
        lora_dropout_p=0.0,
        lora_alpha=1,
        init_loldu_weights="ldu",
        init_method="lu",
    ):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            fan_in_fan_out=False,
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
            layer_module=layer,
            init_loldu_weights="ldu",
            init_method=init_method,
        )

    @classmethod
    def from_conv2d(
        cls,
        layer,
        rank=4,
        lora_dropout_p=0.0,
        lora_alpha=1,
        init_loldu_weights="ldu",
        init_method="lu",
    ):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        # layer_weights = layer
        return cls(
            fan_in,
            fan_out,
            fan_in_fan_out=False,
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
            layer_module=layer,
        )

    @classmethod
    def from_embedding(
        cls,
        layer,
        rank=4,
        lora_dropout_p=0.0,
        lora_alpha=1,
        init_loldu_weights="ldu",
        init_method="lu",
    ):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            fan_in_fan_out=True,
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
            layer_module=layer,
        )

    def disable_loldu(self):
        self.forward_fn = lambda x: x

    def enable_loldu(self):
        self.forward_fn = self.loldu_forward


default_loldu_config = (
    {  # specify which layers to add loldu to, by default only add to linear layers
        nn.Linear: {
            "weight": partial(LoLDUParametrization.from_linear, rank=768),
        },
    }
)


def apply_loldu(layer, register=True, merge=False, loldu_config=default_loldu_config):
    #    这行定义了一个函数`apply_loldu`，它接受一个层（`layer`），三个可选参数`register`（默认为True），
    # `merge`（默认为False），和`loldu_config`（默认为`default_loldu_config`）。
    # merge_loldu : register=False, merge=True
    """add loldu parametrization to a layer, designed to be used with model.apply"""
    if register:  #    这个条件判断是检查是否需要注册LoLDU参数化。
        if (
            type(layer) in loldu_config
        ):  #    如果当前层的类型在`loldu_config`中定义了相应的LoLDU参数化设置，则继续执行。
            # print(loldu_config[type(layer)])#    {'weight': functools.partial(<class '__main__.LoLDUParametrization'>, rank=8)}
            for attr_name, parametrization in loldu_config[type(layer)].items():
                # attr_name:"weight"; parametrization: partial(LoLDUParametrization.from_linear, rank=8)
                parametrize.register_parametrization(
                    layer, attr_name, parametrization(layer)
                )

    else:  # this will remove all parametrizations, use with caution
        #    如果`register`为False，则进入这个分支，这个分支将移除所有参数化。
        if hasattr(layer, "parametrizations"):  #    检查层是否有`parametrizations`属性。
            for attr_name in layer.parametrizations.keys():  #    如果有，遍历所有的参数化属性。
                parametrize.remove_parametrizations(
                    layer, attr_name, leave_parametrized=merge
                )
            #     移除每个参数化，如果`merge`为True，则在移除时保留参数化的结果。


#    定义了一个函数`add_loldu`，接受一个模型和一个可选的LoLDU配置（默认为`default_loldu_config`）。
def add_loldu(model, loldu_config=default_loldu_config):
    """add loldu parametrization to all layers in a model. Calling it twice will add loldu twice"""
    # 给模型中所有层添加LoLDU参数化。如果调用两次，会添加两次LoLDU。
    model.apply(partial(apply_loldu, loldu_config=loldu_config))


#    使用`apply`方法在模型的所有层上应用`apply_loldu`函数。这里用到了`partial`，
# 它创建了一个新的函数，将`loldu_config`作为参数预先填充到`apply_loldu`中。


def add_loldu_by_name(model, target_module_names, loldu_config=default_loldu_config):
    for name, layer in model.named_modules():
        # name: 0,1,2,...
        # layer: Sequential(...
        # for example if target_module_names = ['fc', 'conv'], then all layers whose name contains 'fc' or 'conv' will be added loldu
        if any([m in name for m in target_module_names]):
            add_loldu(layer, loldu_config=loldu_config)


def merge_loldu(model):
    """merge loldu parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_loldu, register=False, merge=True))


def remove_loldu(model):
    """remove loldu parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_loldu, register=False, merge=False))
