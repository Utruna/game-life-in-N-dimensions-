from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSet:
    birth: frozenset[int]
    survive: frozenset[int]
    spec: str

    @classmethod
    def from_spec(cls, spec: str) -> "RuleSet":
        normalized = spec.strip().upper().replace(" ", "")
        try:
            birth_part, survive_part = normalized.split("/")
        except ValueError as exc:
            raise ValueError(f"Invalid rule spec '{spec}'. Expected B.../S...") from exc

        if not birth_part.startswith("B") or not survive_part.startswith("S"):
            raise ValueError(f"Invalid rule spec '{spec}'. Expected B.../S...")

        birth = _parse_numbers(birth_part[1:])
        survive = _parse_numbers(survive_part[1:])
        return cls(birth=frozenset(birth), survive=frozenset(survive), spec=f"B{','.join(map(str, sorted(birth)))}/S{','.join(map(str, sorted(survive)))}")


def _parse_numbers(raw: str) -> set[int]:
    if not raw:
        return set()

    if "," in raw:
        parts = raw.split(",")
    else:
        parts = list(raw)

    numbers: set[int] = set()
    for item in parts:
        if item == "":
            continue
        if not item.isdigit():
            raise ValueError(f"Invalid rule number '{item}'")
        numbers.add(int(item))
    return numbers
