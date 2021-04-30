import torch

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class MaxPool2d(torch.nn.MaxPool2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x):
        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            padding = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
            dilation = (self.dilation, self.dilation) if isinstance(self.dilation, int) else self.dilation
            kernel_size = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
            stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], padding, dilation, kernel_size, stride
                )
            ]
            output_shape = [x.shape[0], x.shape[1]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            return empty

        x = super().forward(x)
        return x

class Conv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super(Conv3d, self).forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() > 0:
            x = super(ConvTranspose2d, self).forward(x)
            if self.norm is not None:
                x = self.norm(x)
            if self.activation is not None:
                x = self.activation(x)
            return x

        # get output shape
        # When input is empty, we want to return a empty tensor with "correct" shape,
        # So that the following operations will not panic
        # if they check for the shape of the tensor.
        # This computes the height and width of the output tensor
        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.out_channels] + output_shape
        # This is to make DDP happy.
        # DDP expects all workers to have gradient w.r.t the same set of parameters.
        _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return _NewEmptyTensorOp.apply(x, output_shape) + _dummy