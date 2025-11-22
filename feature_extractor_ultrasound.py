# Modified feature extractor for ultrasound images
# Based on feature_extractor.py but uses DINOv2 (better for medical images)

import torch
from torchvision import transforms
from typing import Tuple

def get_transform(model_name: str, is_ultrasound: bool = True) -> transforms.Compose:
    """
    Get the appropriate torchvision transform for the given model name.

    Args:
        model_name (str): The name of the model.
        is_ultrasound (bool): Whether input is ultrasound (handles grayscale)

    Returns:
        transforms.Compose: Transformations to apply to the input images.
    """
    if model_name.lower() in ["dino", "dinov2"]:
        transform_list = []

        # Ultrasound-specific preprocessing
        if is_ultrasound:
            # Note: Your ultrasound images are already RGB, but if they were grayscale:
            # transform_list.append(transforms.Grayscale(num_output_channels=3))
            pass

        transform_list.extend([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Use ImageNet normalization (works well for DINOv2)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        return transforms.Compose(transform_list)
    else:
        raise NotImplementedError(f"model_name: {model_name} is not supported")

def get_feat_model(model_name: str) -> Tuple[torch.nn.Module, int]:
    """
    Get the feature extraction model and its output feature size.

    Args:
        model_name (str): The name of the model ("DINO" or "DINOv2").

    Returns:
        Tuple[torch.nn.Module, int]: The feature extraction model and output feature size.
    """
    if model_name.lower() == "dino":
        # Use original DINO ViT-Small (histopathology pre-trained)
        from feature_extractor import vit_small
        model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        return model, 384

    elif model_name.lower() == "dinov2":
        # Use DINOv2 (better for general images including medical)
        # This is pre-trained on a larger, more diverse dataset
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # DINOv2 ViT-Small outputs 384 dimensions
        return model, 384

    else:
        raise NotImplementedError(f"model_name: {model_name} is not supported")

# For backward compatibility
if __name__ == "__main__":
    print("Testing feature extractor...")
    model, feat_dim = get_feat_model("dinov2")
    print(f"Model loaded: {type(model)}")
    print(f"Feature dimension: {feat_dim}")

    transform = get_transform("dinov2", is_ultrasound=True)
    print(f"Transform: {transform}")
