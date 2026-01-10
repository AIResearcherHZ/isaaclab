# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import noise_cfg

##
# Noise as functions.
##


def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for bias on first call and update config parameters
    if isinstance(cfg.bias, torch.Tensor):
        cfg.bias = cfg.bias.to(device=data.device)

    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """
    # 首次调用时预计算并缓存range值
    if not hasattr(cfg, '_n_range') or cfg._n_range is None:
        if isinstance(cfg.n_max, torch.Tensor):
            cfg.n_max = cfg.n_max.to(data.device)
        if isinstance(cfg.n_min, torch.Tensor):
            cfg.n_min = cfg.n_min.to(data.device)
        cfg._n_range = cfg.n_max - cfg.n_min
        cfg._device = data.device
    elif hasattr(cfg, '_device') and cfg._device != data.device:
        # 设备变更时重新计算
        if isinstance(cfg.n_max, torch.Tensor):
            cfg.n_max = cfg.n_max.to(data.device)
        if isinstance(cfg.n_min, torch.Tensor):
            cfg.n_min = cfg.n_min.to(data.device)
        cfg._n_range = cfg.n_max - cfg.n_min
        cfg._device = data.device

    if cfg.operation == "add":
        # 使用empty+uniform_避免rand_like的额外开销，直接原地生成
        noise = torch.empty_like(data).uniform_(0, 1)
        noise.mul_(cfg._n_range).add_(cfg.n_min)
        return data.add(noise)
    elif cfg.operation == "scale":
        noise = torch.empty_like(data).uniform_(0, 1)
        noise.mul_(cfg._n_range).add_(cfg.n_min)
        return data.mul(noise)
    elif cfg.operation == "abs":
        noise = torch.empty_like(data).uniform_(0, 1)
        return noise.mul_(cfg._n_range).add_(cfg.n_min)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """
    # 首次调用时缓存设备信息
    if not hasattr(cfg, '_device') or cfg._device is None:
        if isinstance(cfg.mean, torch.Tensor):
            cfg.mean = cfg.mean.to(data.device)
        if isinstance(cfg.std, torch.Tensor):
            cfg.std = cfg.std.to(data.device)
        cfg._device = data.device
    elif hasattr(cfg, '_device') and cfg._device != data.device:
        if isinstance(cfg.mean, torch.Tensor):
            cfg.mean = cfg.mean.to(data.device)
        if isinstance(cfg.std, torch.Tensor):
            cfg.std = cfg.std.to(data.device)
        cfg._device = data.device

    if cfg.operation == "add":
        # 使用empty+normal_避免randn_like的额外开销
        noise = torch.empty_like(data).normal_(0, 1)
        noise.mul_(cfg.std).add_(cfg.mean)
        return data.add(noise)
    elif cfg.operation == "scale":
        noise = torch.empty_like(data).normal_(0, 1)
        noise.mul_(cfg.std).add_(cfg.mean)
        return data.mul(noise)
    elif cfg.operation == "abs":
        noise = torch.empty_like(data).normal_(0, 1)
        return noise.mul_(cfg.std).add_(cfg.mean)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


##
# Batched noise functions for better GPU efficiency
##


def batched_uniform_noise_add_(
    data: torch.Tensor,
    n_min: float | torch.Tensor,
    n_max: float | torch.Tensor,
) -> torch.Tensor:
    """原地批量添加均匀噪声，适用于已拼接的观测数据。

    Args:
        data: 要添加噪声的数据 (num_envs, total_dim)
        n_min: 噪声最小值
        n_max: 噪声最大值

    Returns:
        添加噪声后的数据（原地修改）
    """
    n_range = n_max - n_min
    noise = torch.empty_like(data).uniform_(0, 1)
    noise.mul_(n_range).add_(n_min)
    return data.add_(noise)


def batched_uniform_noise_add_per_dim_(
    data: torch.Tensor,
    n_mins: torch.Tensor,
    n_maxs: torch.Tensor,
) -> torch.Tensor:
    """原地批量添加均匀噪声，每个维度可以有不同的噪声范围。

    Args:
        data: 要添加噪声的数据 (num_envs, total_dim)
        n_mins: 每个维度的噪声最小值 (total_dim,)
        n_maxs: 每个维度的噪声最大值 (total_dim,)

    Returns:
        添加噪声后的数据（原地修改）
    """
    n_ranges = n_maxs - n_mins
    noise = torch.empty_like(data).uniform_(0, 1)
    noise.mul_(n_ranges).add_(n_mins)
    return data.add_(noise)


##
# Noise models as classes
##


class NoiseModel:
    """Base class for noise models."""

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the bias noise configuration
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the bias term
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape
            # expand bias from (num_envs,1) to (num_envs, num_components)
            self._bias = self._bias.repeat(1, self._num_components)
            # now re-sample that expanded bias in-place
            self.reset()
        return super().__call__(data) + self._bias
