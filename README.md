# LoLDU: Low-Rank Adaptation via Lower-Diag-Upper Decomposition

LoLDU is a cutting-edge Parameter-Efficient Fine-Tuning (PEFT) technique designed to drastically reduce the number of trainable parameters while achieving performance levels comparable to full fine-tuning. This document outlines the steps required to integrate LoLDU into your projects effectively.

For further details, please refer to the paper: https://arxiv.org/pdf/2410.13618

## Table of Contents
1. [Installation](#installation)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

## Installation

To install LoLDU, simply use pip:

```bash
git clone https://github.com/SKDDJ/LoLDU
cd LoLDU
pip install -e .
```

## Key Features

- Significantly reduces the number of trainable parameters (up to 2600 times fewer than regular PEFT methods)
- Maintains performance comparable to full fine-tuning
- Leverages Lower-Diag-Upper Decomposition (LDU) for faster convergence and orthogonality
- Focuses on optimizing the diagonal matrix for scaling transformations
- Compatible with various model architectures (e.g., LLaMA2, Roberta, ViT, Stable Diffusion)

## Quick Start

Here's a quick example of how to use LoLDU:

```python
import torch
import torch.nn as nn
from functools import partial
from minloldu import LoLDUParametrization, add_loldu, get_loldu_params

# Define your model
model = YourModel()

# Define LoLDU configuration
loldu_config = {
    nn.Linear: {
        "weight": partial(LoLDUParametrization.from_linear, rank=15),
    },
}

# Add LoLDU to the model
add_loldu(model, loldu_config=loldu_config)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Enable gradients for LoLDU parameters
for param in get_loldu_params(model):
    param.requires_grad = True

# Now your model is ready for fine-tuning with LoLDU
```

## API Reference

### Main Functions

1. `add_loldu(model, loldu_config)`
   - Adds LoLDU parameterization to the specified model.
   - `model`: The PyTorch model to modify.
   - `loldu_config`: Configuration dictionary for LoLDU.

2. `get_loldu_params(model, print_shapes=False)`
   - Returns the LoLDU parameters of the model.
   - `model`: The PyTorch model with LoLDU.
   - `print_shapes`: If True, prints the shapes of LoLDU parameters.

3. `disable_loldu(model)`
   - Temporarily disables LoLDU in the model.

4. `enable_loldu(model)`
   - Re-enables LoLDU in the model after disabling.

5. `remove_loldu(model)`
   - Completely removes LoLDU from the model.

6. `merge_loldu(model)`
   - Merges LoLDU parameters into the original model weights for efficient inference.

7. `get_loldu_state_dict(model)`
   - Returns the state dictionary of LoLDU parameters for saving.

### LoLDUParametrization Class

- `LoLDUParametrization.from_linear(layer, rank)`
  - Creates LoLDU parameterization for a linear layer.
  - `layer`: The linear layer to parameterize.
  - `rank`: The rank for the low-rank approximation.

## Usage Examples

### Adding LoLDU to a Model

```python
from minloldu import LoLDUParametrization, add_loldu
from functools import partial

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=15, out_features=15),
        )
    def forward(self, x):
        return self.model(x)

model = MyModel()

loldu_config = {
    nn.Linear: {
        "weight": partial(LoLDUParametrization.from_linear, rank=15),
    },
}

add_loldu(model, loldu_config=loldu_config)
```

### Training with LoLDU

```python
from minloldu import get_loldu_params

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Enable gradients for LoLDU parameters
for param in get_loldu_params(model):
    param.requires_grad = True

# Your training loop here
```

### Saving and Loading LoLDU State

```python
from minloldu import get_loldu_state_dict

# Save LoLDU state
state_dict_to_save = get_loldu_state_dict(model)
torch.save(state_dict_to_save, "loldu_state.pth")

# Load LoLDU state
loaded_state = torch.load("loldu_state.pth")
model.load_state_dict(loaded_state, strict=False)
```

### Merging LoLDU for Inference

```python
from minloldu import merge_loldu

# After training, merge LoLDU for efficient inference
merge_loldu(model)
```

## Best Practices

1. **Choose Appropriate Rank**: The rank parameter in LoLDUParametrization affects the trade-off between parameter efficiency and model performance. Experiment with different ranks to find the optimal balance for your task.

2. **Fine-tune Hyperparameters**: LoLDU may require different learning rates compared to full fine-tuning. Adjust your learning rate and other hyperparameters accordingly.

3. **Monitor Training**: Keep an eye on the training process to ensure that LoLDU is effectively adapting the model. Use validation sets to prevent overfitting.

4. **Merge for Inference**: Always use `merge_loldu()` before deploying your model for inference to eliminate any computational overhead.

5. **Combine with Other Techniques**: LoLDU can be combined with other optimization techniques like quantization for even greater efficiency.

For more detailed information and advanced usage, please refer to the original paper and the source code repository.

---

**Note:**  
Please be aware that this code may not fully replicate the results presented in the paper due to possible human errors that occurred during the preparation and cleaning of the code before its release. If you experience any challenges in reproducing our findings, do not hesitate to reach out to us. Furthermore, we are committed to conducting sanity-check experiments in the near future.

**Acknowledgment**  
Our LoLDU implementation was greatly enhanced by the [minLoRA](https://github.com/cccntu/minLoRA)  codebase.

**BibTeX**  
```bibtex
@misc{shi2024loldulowrankadaptationlowerdiagupper,
  title={LoLDU: Low-Rank Adaptation via Lower-Diag-Upper Decomposition for Parameter-Efficient Fine-Tuning}, 
  author={Yiming Shi and Jiwei Wei and Yujia Wu and Ran Ran and Chengwei Sun and Shiyuan He and Yang Yang},
  year={2024},
  eprint={2410.13618},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2410.13618},
}
```

