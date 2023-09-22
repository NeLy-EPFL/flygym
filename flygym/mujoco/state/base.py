from abc import ABC, abstractmethod
from typing import Any


class BaseState(ABC):
    """Base class for animal state (eg. pose) representations. Behaves
    like a dictionary."""

    @abstractmethod
    def __iter__(self):
        """Returns an iterator over the keys of the state."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Returns the value of a variable in the state given a key."""
        raise NotImplementedError