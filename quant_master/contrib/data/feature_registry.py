# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metadata and family-level selection for model input features."""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


FeatureWindow = Optional[Union[int, Tuple[int, ...]]]
FamilySelection = Union[str, Iterable[str]]


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _family_set(families: FamilySelection) -> set:
    values = (families,) if isinstance(families, str) else families
    return {_require_text(value, "family") for value in values}


@dataclass(frozen=True)
class FeatureSpec:
    """A concrete feature column and its point-in-time metadata.

    ``descriptor`` is the expression consumed by ``QuantMasterDataLoader``.
    ``available_lag`` is the minimum number of trading sessions between the
    source observation and the feature becoming available to a daily model.
    ``window`` may contain multiple values for features such as a 5/20-day
    crossover.
    """

    name: str
    family: str
    descriptor: str
    window: FeatureWindow
    data_requirements: Tuple[str, ...]
    available_lag: int
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "name"))
        object.__setattr__(self, "family", _require_text(self.family, "family"))
        object.__setattr__(self, "descriptor", _require_text(self.descriptor, "descriptor"))
        object.__setattr__(self, "source", _require_text(self.source, "source"))

        if isinstance(self.data_requirements, str):
            raise TypeError("data_requirements must be an iterable of field names, not a string")
        requirements = tuple(_require_text(value, "data requirement") for value in self.data_requirements)
        if not requirements:
            raise ValueError("data_requirements must contain at least one field")
        if len(requirements) != len(set(requirements)):
            raise ValueError("data_requirements must not contain duplicates")
        object.__setattr__(self, "data_requirements", requirements)

        if isinstance(self.available_lag, bool) or not isinstance(self.available_lag, int):
            raise TypeError("available_lag must be an integer number of trading sessions")
        if self.available_lag < 0:
            raise ValueError("available_lag must be non-negative")

        window = self.window
        if isinstance(window, list):
            window = tuple(window)
            object.__setattr__(self, "window", window)
        if window is not None:
            windows = (window,) if isinstance(window, int) and not isinstance(window, bool) else window
            if not isinstance(windows, tuple) or not windows:
                raise TypeError("window must be an integer, a non-empty tuple of integers, or None")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in windows):
                raise TypeError("window values must be integers")
            if any(value < 0 for value in windows):
                raise ValueError("window values must be non-negative")


class FeatureRegistry:
    """Ordered registry of uniquely named feature columns."""

    def __init__(self, features: Iterable[FeatureSpec] = ()) -> None:
        self._features: Dict[str, FeatureSpec] = {}
        self.register_many(features)

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self._features.values())

    def __contains__(self, name: object) -> bool:
        return name in self._features

    @property
    def families(self) -> Tuple[str, ...]:
        """Return family names in first-registration order."""

        return tuple(dict.fromkeys(feature.family for feature in self._features.values()))

    def family_counts(self) -> Dict[str, int]:
        counts = {family: 0 for family in self.families}
        for feature in self._features.values():
            counts[feature.family] += 1
        return counts

    def get(self, name: str) -> FeatureSpec:
        return self._features[name]

    def register(self, feature: FeatureSpec) -> FeatureSpec:
        """Register one feature and return it for convenient inline use."""

        self.register_many((feature,))
        return feature

    def register_many(self, features: Iterable[FeatureSpec]) -> None:
        """Register features atomically after validating all names."""

        pending = list(features)
        if any(not isinstance(feature, FeatureSpec) for feature in pending):
            raise TypeError("all registered features must be FeatureSpec instances")

        pending_names = [feature.name for feature in pending]
        duplicates = {name for name, count in Counter(pending_names).items() if count > 1}
        duplicates.update(name for name in pending_names if name in self._features)
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate feature name(s): {joined}")

        self._features.update((feature.name, feature) for feature in pending)

    def select(
        self,
        families: Optional[FamilySelection] = None,
        exclude_families: FamilySelection = (),
    ) -> Tuple[FeatureSpec, ...]:
        """Select features by family while preserving registration order."""

        included = None if families is None else _family_set(families)
        excluded = _family_set(exclude_families)
        return tuple(
            feature
            for feature in self._features.values()
            if (included is None or feature.family in included) and feature.family not in excluded
        )

    def ablate(self, families: FamilySelection) -> "FeatureRegistry":
        """Return an independent registry with the requested families removed."""

        return FeatureRegistry(self.select(exclude_families=families))

    def expand_columns(
        self,
        families: Optional[FamilySelection] = None,
        exclude_families: FamilySelection = (),
    ) -> Tuple[List[str], List[str]]:
        """Expand selected metadata into loader-compatible fields and names."""

        selected = self.select(families=families, exclude_families=exclude_families)
        return [feature.descriptor for feature in selected], [feature.name for feature in selected]


__all__: Sequence[str] = ("FamilySelection", "FeatureRegistry", "FeatureSpec", "FeatureWindow")
