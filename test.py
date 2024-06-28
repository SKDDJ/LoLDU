import torch
import torch.nn as nn
from functools import partial

from minloldu import (
    LoLDUParametrization,
    add_loldu,
    apply_to_loldu,
    disable_loldu,
    enable_loldu,
    get_loldu_params,
    merge_loldu,
    name_is_loldu,
    remove_loldu,
    get_loldu_state_dict,
)

_ = torch.set_grad_enabled(False)

# import sys
# sys.setrecursionlimit(1000000)  # 举例增加到1500，根据实际需要调整


# a simple model


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 正确地将模型定义为类的属性
        self.model = nn.Sequential(
            nn.Linear(in_features=15, out_features=15),
            # nn.ReLU(),  # 可选：添加一个非线性激活层以提升模型的表达能力
            # nn.Linear(in_features=70, out_features=),
        )

    def forward(self, x):
        # 定义前向传播
        return self.model(x)

    # def __repr__(self):
    #     # 返回模型的简化字符串表示
    #     return "<MyModel with 2 layers>"


model = MyModel()

x = torch.randn(1, 15)
print("The RANDOM x", x)

y = model(x)
print("original y is", y)  # original y is tensor([[ 0.1539, -0.4083, -0.3811]])
# Y0 = y

loldu_config = {
    nn.Linear: {
        "weight": partial(LoLDUParametrization.from_linear, rank=15),
    },
}

# add loldu to the model
# becase B is initialized to 0, the output is the same as before

# import pysnooper
# with pysnooper.snoop():
add_loldu(model, loldu_config=loldu_config)
y = model(x)
print("y after add loldu", y)  # y after add lora tensor([[ 0.2840, -0.3440, -0.4243]])
# print("just show the code runs here")
# print(model)  # <MyModel with 2 layers>
# for name, module in model._modules.items():
#     print(name, module.__class__.__name__)
# 0 ParametrizedLinear
# 1 ParametrizedLinear
# def name_is_loldu(name):
#     # print("name_is_loldu")
#     # print(name.split("."))
#     """['0', 'bias']
#     ['0', 'parametrizations', 'weight', 'original']
#     ['0', 'parametrizations', 'weight', '0', 'lora_A']
#     ['0', 'parametrizations', 'weight', '0', 'lora_B']
#     ['0', 'parametrizations', 'weight', '0', 'vector_z']
#     ['1', 'bias']
#     ['1', 'parametrizations', 'weight', 'original']
#     ['1', 'parametrizations', 'weight', '0', 'lora_A']
#     ['1', 'parametrizations', 'weight', '0', 'lora_B']
#     ['1', 'parametrizations', 'weight', '0', 'vector_z']
#     """
#     return (
#         len(name.split(".")) >= 4
#         and (name.split(".")[-4]) == "parametrizations"
#         # and name.split(".")[-1] in ["vector"]
#         and name.split(".")[-1] in ["lora_A", "lora_B","vector_z"]
#     )
# for n, p in model.named_parameters():
# helo = name_is_loldu(n)
# print(n)
# print("\n")
# print(helo)
# if  name_filter(n):
# if print_shapes:

# print(n, p.shape)
# """0.bias torch.Size([7])
# 0.parametrizations.weight.original torch.Size([7, 5])
# 0.parametrizations.weight.0.lora_A torch.Size([2, 5])
# 0.parametrizations.weight.0.lora_B torch.Size([7, 2])
# 0.parametrizations.weight.0.vector_z torch.Size([2])
# 1.bias torch.Size([3])
# 1.parametrizations.weight.original torch.Size([3, 7])
# 1.parametrizations.weight.0.lora_A torch.Size([2, 7])
# 1.parametrizations.weight.0.lora_B torch.Size([3, 2])
# 1.parametrizations.weight.0.vector_z torch.Size([2])
# """


# aaa = get_loldu_params(model, print_shapes=True)

# for item in aaa:
# print the trainable params
#     print(item)

# print("try print model again")
# import pdb

# Assuming 'model' is defined somewhere above this line

# pdb.set_trace()  # This line will initiate the debugger

# Once the debugger is active, you can use commands like 'p model'
# to print the model or 'p dir(model)' to see its attributes.
# Be cautious with 'print(model)' if it's causing a recursion issue.

# Assuming you've examined 'model' or made necessary adjustments, and want to try printing again
# try:
#     print(model) # <MyModel with 2 layers>
# except RecursionError as e:
#     print("RecursionError encountered: ", e)

# To exit the debugger, you can use the 'c' command to continue execution, or 'q' to quit the debugger.

# # aaa is a generator and I want to use a loop to check the aaa
# for i in aaa:
#     print(i)

# print(model)
# from labml.logger import inspect
# inspect(model)
# from torchkeras import summary
# summary(model, input_shape=(5,))
# assert torch.allclose(y, Y0)


# to make the output different, we need to initialize B to something non-zero
# model.apply(apply_to_loldu(lambda x: torch.nn.init.ones_(x.lora_B)))
# y = model(x)
# print(y)
# assert not torch.allclose(y, Y0)
# Y1 = y
# print(model)


# now let's try to disable loldu, the output is the same as before loldu is added
disable_loldu(model)
# print(model)
# y = model(x)
print(
    "y after disable loldu", y
)  # y after disable loldu tensor([[ 0.1539, -0.4083, -0.3811]])
print("end" * 20)
exit()
y = model(x)
assert torch.allclose(y, Y0)


# enable loldu again
enable_loldu(model)
y = model(x)
assert torch.allclose(y, Y1)


# let's save the state dict for later use
state_dict_to_save = get_loldu_state_dict(model)
state_dict_to_save.keys()

# you can remove loldu from the model
remove_loldu(model)


# lets try to load the loldu back
# first we need to add loldu to the model
add_loldu(model)
# then we can load the loldu parameters
# strict=False is needed because we are loading a subset of the parameters
_ = model.load_state_dict(state_dict_to_save, strict=False)
y = model(x)
assert torch.allclose(y, Y1)


# we can merge it to make it a normal linear layer, so there is no overhead for inference
merge_loldu(model)
y = model(x)
assert torch.allclose(y, Y1)


# model now has no loldu parameters
print(model)
