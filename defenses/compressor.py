try:
    import cupy as xp
except ImportError:
    import numpy as xp
import torch

# import numpy as np

def identity(x, *args, **kwargs):
    return x

# top_a
def top(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a >= dim:
        return x
    index_array = xp.argpartition(x, kth=a, axis=0)[a:]
    xp.put_along_axis(x, index_array, 0, axis=0)
    return x

# x = np.random.randint(0, 100, 24).reshape(6, 4)
# x
# top(x, 2)

def random(x, a):
    x = x.clone()  # Prevent modifying input tensor
    dim = x.shape[0]

    if a == 0:
        return torch.zeros_like(x)
    if a == dim:
        return x

    if x.ndim == 2:
        for i in range(x.shape[1]):
            zero_mask = torch.randperm(dim, device=x.device)[:dim - a]
            x[zero_mask, i] = 0
    else:
        zero_mask = torch.randperm(dim, device=x.device)[:dim - a]
        x[zero_mask] = 0

    return x






# gsgd_b
def gsgd(x, b):
    norm = xp.linalg.norm(x, axis=0)
    return norm / (2 ** (b - 1)) * xp.sign(x) * xp.floor(
                (2 ** (b - 1)) / norm * xp.abs(x) + xp.random.uniform(0, 1, x.shape)
            )


# random quantization 2-norm with level s
def random_quantization(x, s):
    dim = x.shape[0]
    xnorm = xp.linalg.norm(x)
    if s == 0 or xnorm == 0:
        return xp.zeros(dim, dtype=int)
    noise = xp.random.uniform(0, 1, dim)
    rounded = xp.floor(s * xp.abs(x) / xnorm + noise)
    compressed = (xnorm / s) * xp.sign(x) * rounded
    return compressed


# natural compression (power of 2 for each coordinate)
def natural_compression(gradients):
    """
    Compresses a list of PyTorch tensors using the natural compression algorithm.
    """
    compressed_gradients = []
    
    for grad in gradients:
        if isinstance(grad, torch.Tensor):
            grad = grad.cpu().numpy()  # Ensure conversion to NumPy array
        
        dim = grad.shape[0]
        logx = xp.ma.log2(xp.abs(grad)).filled(-15)
        logx_floor = xp.floor(logx)
        noise = xp.random.uniform(0.0, 1.0, grad.shape)
        leftx = xp.exp2(logx_floor)
        rounded = xp.floor(xp.ma.log2(xp.abs(grad) + leftx * noise).filled(-15))
        compressed = xp.sign(grad) * xp.exp2(rounded)
        
        compressed_gradients.append(torch.tensor(compressed, dtype=torch.float32))
    
    return compressed_gradients