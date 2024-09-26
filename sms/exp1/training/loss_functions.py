import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional    

def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    # used to compute the covariance loss
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(
        projected_batch,
        weight_inv,
        weight_var,
        weight_cov
        ):
    """
    Implementation of the VICReg loss function.
    For a batch of anchors and positives passed through the projection head, compute the VICReg loss.
    The loss is the sum of three terms:
    - The invariance loss (MSE between the expanded anchors and positives)
    - The variance loss (encourages the variance of the embeddings to be large)
    - The covariance loss (encourages the off-diagonal elements of the covariance matrix to be zero)
    (taken from https://github.com/Irislucent/motif-encoder)
    """
    projected_anchors = projected_batch[0]
    projected_positives = projected_batch[1]

    loss_inv = F.mse_loss(projected_anchors, projected_positives)

    # the variance inside a batch is encouraged to be large (close to 1)
    anchors_resid = projected_anchors - projected_anchors.mean(dim=0)
    positives_resid = projected_positives - projected_positives.mean(dim=0)

    anchors_std = torch.sqrt(anchors_resid.var(dim=0) + 0.0001)
    positives_std = torch.sqrt(positives_resid.var(dim=0) + 0.0001)

    loss_var = torch.mean(F.relu(1 - anchors_std)) / 2 + torch.mean(F.relu(1 - positives_std)) / 2

    # the off-diagonal coefficients in the cov matrix are pushed to 0 to decorrelate the dimensions of the embeddings
    anchors_cov = anchors_resid.T @ anchors_resid / (anchors_resid.shape[0] - 1)
    positives_cov = positives_resid.T @ positives_resid / (positives_resid.shape[0] - 1)
    loss_cov = _off_diagonal(anchors_cov).pow_(2).sum().div(anchors_cov.shape[1]) \
        + _off_diagonal(positives_cov).pow_(2).sum().div(positives_cov.shape[1])

    # print(f"projected_anchors requires_grad: {projected_anchors.requires_grad}")
    # print(f"projected_positives requires_grad: {projected_positives.requires_grad}")

    # total loss
    train_loss = loss_inv * weight_inv  \
        + loss_var * weight_var \
        + loss_cov * weight_cov
    # print(f"train_loss requires_grad: {train_loss.requires_grad}")
    return train_loss, loss_inv, loss_var, loss_cov

def contrastive_loss(
        anchor,
        positive,
        negative,
        temperature=0.1
        ):
    """
    Implementation of the contrastive loss function using Euclidean distance.
    
    Args:
        anchor (torch.Tensor): The anchor embeddings.
        positive (torch.Tensor): The positive embeddings.
        negative (torch.Tensor): The negative embeddings.
        temperature (float): Temperature parameter to scale the loss.
    
    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    # Compute Euclidean distances
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    
    # Compute exponentials
    exp_pos = torch.exp(-pos_dist / temperature)
    exp_neg = torch.exp(-neg_dist / temperature)
    
    # Compute loss
    loss = -torch.log(exp_pos / (exp_pos + exp_neg))
    
    return loss.mean()

def triplet_loss(
        anchor,
        positive,
        negative,
        margin=1.0
        ):
    """
    Implementation of the triplet loss function using Euclidean distance.
    
    Args:
        anchor (torch.Tensor): The anchor embeddings.
        positive (torch.Tensor): The positive embeddings.
        negative (torch.Tensor): The negative embeddings.
        margin (float): The margin for the triplet loss.
    
    Returns:
        torch.Tensor: The computed triplet loss.
    """
    # Compute Euclidean distances
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    
    # Compute triplet loss
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    
    return loss.mean()