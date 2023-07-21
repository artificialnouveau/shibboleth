# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:16:14 2023

@author: AZhuparris
"""

import torch
import io
import torch.serialization
import os


def load_model_from_file(file_path, map_location='cpu'):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise ValueError(f"No such file: '{file_path}'")

    try:
        with open(file_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
    except Exception as e:
        raise IOError(f"Failed to read the file '{file_path}': {str(e)}")

    model = torch.load(buffer, map_location=map_location)
    return model


def interpolate_models(model_path1, model_path2, output_path, alpha):
    device = torch.device('cpu')  # use 'cuda' if you have a GPU available and CUDA installed

    # Load the models
    model1 = load_model_from_file(model_path1, map_location=device)
    model2 = load_model_from_file(model_path2, map_location=device)

    # Ensure the models have the same architecture
    assert model1.keys() == model2.keys(), "Models don't have the same architecture"

    # Interpolate the weights
    interpolated_model = {}
    for param_name in model1.keys():
        interpolated_model[param_name] = (
            1 - alpha) * model1[param_name] + alpha * model2[param_name]

    # Save the new model
    torch.save(interpolated_model, output_path)


# Usage
for alphas in range(0, 10):
    interpolate_models('./checkpoints/effi_checkpoint_160000.pth',
                       './checkpoints/amir_checkpoint_300000.pth',
                       './checkpoints/interpolated_model_alpha_'+str(alphas)+'.pth', alpha=alphas)
