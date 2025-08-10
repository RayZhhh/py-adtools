```python
import numpy as np, numba
import torch.nn as nn

dataset_loader = ...


@numba.jit()
def load_dataset(path, size) -> np.ndarray:
    """Load dataset from local directory."""
    ...

@some.decorators    
class CNN(nn.Module):
    """My CNN."""
    
    class ResidualBlcok(nn.Module):
		...
        
    def __init__():
        super().__init__()
        ...
    
    @some.decorators
    def forward(x):
        ...
```

