#!/usr/bin/python
"""Per-task LoRA specialist library + served generator (extends the AHE-3.1 substrate).

The "modular specialization + routing" the SAI paper calls for, on the weight
side: a registry of ``task_signature → trained LoRA adapter`` so *many* specialists
coexist on one shared base model and are routed per task, one specialist per task
signature, each keyed to its served adapter name.

:func:`served_generator` bridges a served adapter to the SAI factory (AHE-3.29):
it returns a ``GenerateFn`` that produces candidates from a specific LoRA adapter
via the inference backend's per-request adapter swap — i.e. the factory's weight
arm, once it has trained + served an adapter, hands the controller a generator
bound to that specialist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpecialistAdapter:
    """One trained LoRA specialist for a task."""

    task_signature: str
    adapter_id: str
    base_model: str
    checkpoint_path: str | None = None


class AdapterLibrary:
    """A growing registry of per-task LoRA specialists (one base, many adapters)."""

    def __init__(self) -> None:
        self._by_task: dict[str, SpecialistAdapter] = {}

    def register(
        self,
        task_signature: str,
        adapter_id: str,
        base_model: str,
        *,
        checkpoint_path: str | None = None,
        base_url: str | None = None,
        tier: str = "medium",
    ) -> SpecialistAdapter:
        """Add (or replace) the specialist for ``task_signature``"""
        spec = SpecialistAdapter(
            task_signature=task_signature,
            adapter_id=adapter_id,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
        )
        self._by_task[task_signature] = spec
        return spec

    def resolve(self, task_signature: str) -> SpecialistAdapter | None:
        """The specialist adapter for a task, or ``None`` if none trained yet."""
        return self._by_task.get(task_signature)

    def task_signatures(self) -> list[str]:
        return list(self._by_task)

    def __len__(self) -> int:
        return len(self._by_task)


def served_generator(
    backend: Any,
    adapter_id: str,
    *,
    scaffold_to_prompt: Callable[[str], str] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Callable[[str], str]:
    """A SAI-factory ``GenerateFn`` that serves candidates from a LoRA ``adapter_id``.

    Calls the inference backend with the per-request ``adapter`` swap, so the
    factory generates candidates from a specific trained specialist on the shared
    base server (no model reload between specialists).
    """

    def generate(scaffold: str) -> str:
        prompt = scaffold_to_prompt(scaffold) if scaffold_to_prompt else scaffold
        out = backend.generate(
            prompt,
            n=1,
            max_tokens=max_tokens,
            temperature=temperature,
            adapter=adapter_id,
        )
        return out[0]["text"] if out else ""

    return generate
