from torch import Tensor
from typing import Any


class Element:
    _on_errors_ = {"omit", "raise"}
    _attr_method_ = "get_attr"
    _attr_script_ = "attr_script"
    _tensor_method_ = "get_tensor"
    _tensor_script_ = "tensor_script"
    _update_method_ = "get_update"
    _update_script_ = "update_script"

    def __init__(self, mutable: bool = True, on_error: str = "omit", **kwargs) -> None:
        assert (
            on_error in self.__class__._on_errors_
        ), f"Invalid on-error policy, must be one of {self.__class__._on_errors_}"

        self.mutable = mutable
        self._on_error = on_error
        self.__dict__.update(kwargs)
    
    @property
    def name(self):
        return self.__class__.name
    
    def __eq__(self, value: object) -> bool:
        return self.name == value.name

    def __hash__(self) -> int:
        return self.name.__hash__()

    @property
    def _attr(self) -> str | None:
        method = callable(getattr(self, self.__class__._attr_method_, None))
        if method:
            return self.__class__._attr_method_
        script = isinstance(getattr(self, self.__class__._attr_script_, None), str)
        if script:
            return self.__class__._attr_script_
        return None

    @property
    def _tensor(self) -> str | None:
        method = callable(getattr(self, self.__class__._tensor_method_, None))
        if method:
            return self.__class__._tensor_method_
        script = isinstance(getattr(self, self.__class__._tensor_script_, None), str)
        if script:
            return self.__class__._tensor_script_
        return None

    @property
    def _update(self) -> str | None:
        method = callable(getattr(self, self.__class__._update_method_, None))
        if method:
            return self.__class__._update_method_
        script = isinstance(getattr(self, self.__class__._update_script_, None), str)
        if script:
            return self.__class__._update_script_
        return None

    @property
    def attr(self, **kwargs) -> Any:
        method = self._attr
        if not method:
            if self._on_error == "omit":
                # TODO log error
                return None
            elif self._on_error == "raise":
                raise AttributeError(
                    f"Either `{self.__class__._attr_method_}` or `{self.__class__._attr_script_}` must be provided when computing attributes"
                )
            else:
                raise ValueError(
                    f"{self._on_error} is not a valid on-error policy, must be one of {self.__class__._on_errors_}"
                )

        if method == self.__class__._attr_method_:
            return self.get_attr(**kwargs)

        elif method == self.__class__._attr_script_:
            raise NotImplementedError("Lambda script compiler is not implemented yet")

    @property
    def tensor(self, **kwargs) -> Tensor | None:
        method = self._tensor
        if not method:
            if self._on_error == "omit":
                # TODO log error
                return None
            elif self._on_error == "raise":
                raise AttributeError(
                    f"Either `{self.__class__._tensor_method_}` or `{self.__class__._tensor_script_}` must be provided when computing tensors"
                )
            else:
                raise ValueError(
                    f"{self._on_error} is not a valid on-error policy, must be one of {self.__class__._on_errors_}"
                )

        if method == self.__class__._tensor_method_:
            return self.get_tensor(**kwargs)

        elif method == self.__class__._tensor_script_:
            raise NotImplementedError("Lambda script compiler is not implemented yet")

    def update(self, **kwargs) -> bool:
        method = self._update
        if not (method or self.mutable):
            if self._on_error == "omit":
                # TODO log error
                return False
            elif self._on_error == "raise":
                if not self.mutable:
                    raise AttributeError(
                    "Immutable elements cannot be updated"
                )
                raise AttributeError(
                    f"Either `{self.__class__._update_method_}` or `{self.__class__._update_script_}` must be provided when computing updates"
                )
            else:
                raise ValueError(
                    f"{self._on_error} is not a valid on-error policy, must be one of {self.__class__._on_errors_}"
                )

        if method == self.__class__._update_method_:
            updates = self.get_update(**kwargs)
            self.__dict__.update(updates)
            return True

        elif method == self.__class__._update_script_:
            raise NotImplementedError("Lambda script compiler is not implemented yet")
        return False
