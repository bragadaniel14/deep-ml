import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        # TODO: implement your custom transformation logic here
        noise = torch.randn_like(x) * 0.05
        x = x+noise
        mean = torch.mean(x)
        var = torch.std(x)+ 1e-5
        x = (x-mean)/var
        return x
