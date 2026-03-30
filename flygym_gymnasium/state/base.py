from abc import ABC, abstractmethod
from typing import Any


class BaseState(ABC):
    """Base class for animal state (e.g. pose) representations. Behaves
    like a dictionary."""

    @abstractmethod
    def __iter__(self, *args, **kwargs):
        """Returns an iterator over the keys of the state."""
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Returns the value of a variable in the state given a key. This
        is to be used in the style of dictionaries: e.g.
        ``state["RFCoxa_angle"]``.
        """
        pass
