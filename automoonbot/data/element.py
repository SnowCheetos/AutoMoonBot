from typing import Any
from torch import Tensor
from functools import lru_cache


class Element:
    _on_errors_ = {"omit", "raise"}
    _attr_method_ = "get_attr"
    _attr_script_ = "attr_script"
    _tensor_method_ = "get_tensor"
    _tensor_script_ = "tensor_script"
    _update_method_ = "get_update"
    _update_script_ = "update_script"

    def __init__(
        self,
        mutable: bool = True,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        if on_error not in self.__class__._on_errors_:
            raise ValueError(
                f"Invalid on-error policy, must be one of {self.__class__._on_errors_}"
            )

        self.mutable = mutable
        self._on_error = on_error
        self.__dict__.update(kwargs)

    @property
    def name(self) -> str:
        return self.__class__.name

    def __eq__(self, value: object) -> bool:
        return self.name == value.name

    def __hash__(self) -> int:
        return self.name.__hash__()

    @property
    @lru_cache(maxsize=None)
    def _attr(self) -> str | None:
        if callable(getattr(self, self.__class__._attr_method_, None)):
            return self.__class__._attr_method_
        if isinstance(getattr(self, self.__class__._attr_script_, None), str):
            return self.__class__._attr_script_
        return None

    @property
    @lru_cache(maxsize=None)
    def _tensor(self) -> str | None:
        if callable(getattr(self, self.__class__._tensor_method_, None)):
            return self.__class__._tensor_method_
        if isinstance(getattr(self, self.__class__._tensor_script_, None), str):
            return self.__class__._tensor_script_
        return None

    @property
    @lru_cache(maxsize=None)
    def _update(self) -> str | None:
        if callable(getattr(self, self.__class__._update_method_, None)):
            return self.__class__._update_method_
        if isinstance(getattr(self, self.__class__._update_script_, None), str):
            return self.__class__._update_script_
        return None

    @property
    def attr(self, **kwargs) -> Any | None:
        method = self._attr
        if not method:
            self._handle_error(
                f"Either `{self.__class__._attr_method_}` or `{self.__class__._attr_script_}` must be provided when computing attributes"
            )
            return None

        if method == self.__class__._attr_method_:
            return self.get_attr(**kwargs)
        elif method == self.__class__._attr_script_:
            raise NotImplementedError("Lambda script compiler is not implemented yet")

    @property
    def tensor(self, **kwargs) -> Tensor | None:
        method = self._tensor
        if not method:
            self._handle_error(
                f"Either `{self.__class__._tensor_method_}` or `{self.__class__._tensor_script_}` must be provided when computing tensors"
            )
            return None

        if method == self.__class__._tensor_method_:
            return self.get_tensor(**kwargs)
        elif method == self.__class__._tensor_script_:
            raise NotImplementedError("Lambda script compiler is not implemented yet")

    def update(self, **kwargs) -> bool:
        method = self._update
        if method is None or not self.mutable:
            self._handle_error(
                f"Either `{self.__class__._update_method_}` or `{self.__class__._update_script_}` must be provided when computing updates",
                mutable=self.mutable,
            )
            return False

        if method == self.__class__._update_method_:
            updates = self.get_update(**kwargs)
            if updates:
                self.__dict__.update(updates)
                return True
            return False
        elif method == self.__class__._update_script_:
            raise NotImplementedError("Lambda script compiler is not implemented yet")
        return False

    def _handle_error(self, message: str, mutable: bool = True) -> None:
        if self._on_error == "omit":
            # TODO: Log error
            return
        elif self._on_error == "raise":
            if not mutable:
                raise AttributeError("Immutable elements cannot be updated")
            raise AttributeError(message)
        else:
            raise ValueError(
                f"{self._on_error} is not a valid on-error policy, must be one of {self.__class__._on_errors_}"
            )

    # Placeholder methods for demonstration purposes
    def get_attr(self, **kwargs) -> None:
        self._handle_error("Subclasses should implement this method")

    def get_tensor(self, **kwargs) -> None:
        self._handle_error("Subclasses should implement this method")

    def get_update(self, **kwargs) -> None:
        self._handle_error("Subclasses should implement this method")
