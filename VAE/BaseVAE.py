from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from abc import abstractmethod
from torch import Tensor



class BaseVae(nn.Module):

    def __init__(self) -> None:
        super(BaseVae, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplemented

    def decode(self, input: Tensor) -> Any:
        raise NotImplemented

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplemented

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

