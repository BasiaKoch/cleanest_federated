from .simple_cnn import SimpleCNN
from .resnet import get_resnet18


def get_model(model_name: str, in_channels: int, num_classes: int, image_size: int = 28, pretrained: bool = False):
    if model_name == 'simple_cnn':
        return SimpleCNN(in_channels, num_classes)
    if model_name == 'resnet18':
        return get_resnet18(in_channels, num_classes, pretrained=pretrained, image_size=image_size)
    raise ValueError(f'Unknown model: {model_name}')


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    import torch

    for name, image_size in [('simple_cnn', 28), ('resnet18', 28), ('resnet18', 64)]:
        model = get_model(name, in_channels=3, num_classes=7, image_size=image_size)
        out = model(torch.randn(1, 3, image_size, image_size))
        assert out.shape == (1, 7)
        print(name, image_size, count_parameters(model))
