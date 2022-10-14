import abc
from typing import List, Optional, Union

import torch
import torch.nn as nn
from hmm_replay.networks import base_network


class Teacher(base_network.BaseNetwork, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        forward_hidden_scaling: float,
        forward_scaling: float,
        nonlinearity: str,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        noise_std: Union[float, int],
        initialisation_std: Optional[float] = None,
        zero_head: bool = False,
    ) -> None:
        self._unit_norm_teacher_head = unit_norm_teacher_head
        self._noise_std = noise_std
        self._zero_head = zero_head

        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            forward_hidden_scaling=forward_hidden_scaling,
            forward_scaling=forward_scaling,
            nonlinearity=nonlinearity,
            weight_normalisation=weight_normalisation,
            initialisation_std=initialisation_std,
        )

        if self._noise_std > 0:
            self._noisy = True
            self._noise_module = torch.distributions.normal.Normal(
                loc=0, scale=self._noise_std
            )
        else:
            self._noisy = False

        self._freeze()

    @property
    def head(self) -> nn.Linear:
        return self._head

    def _freeze(self) -> None:
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False

    def _construct_output_layers(self):
        self._head = nn.Linear(
            self._hidden_dimensions[-1], self._output_dimension, bias=self._bias
        )
        if self._unit_norm_teacher_head:
            head_norm = torch.norm(self._head.weight)

            normalised_head = self._head.weight / head_norm

            self._head.weight.data = normalised_head
        else:
            self._initialise_weights(self._head)

        if self._zero_head:
            self._head.weight.data = torch.zeros_like(self._head.weight.data)

        for param in self._head.parameters():
            param.requires_grad = False

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through head."""
        y = self._head(x)
        if self._noisy:
            y += self._noise_module.sample(y.shape)
        return y
