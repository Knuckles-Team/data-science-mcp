#!/usr/bin/python
"""Per-task LoRA specialist library + served generator (extends the AHE-3.1 substrate).

The "modular specialization + routing" the SAI paper calls for, on the weight
side: a registry of ``task_signature → trained LoRA adapter`` so *many* specialists
coexist on one shared base model and are routed per task — instead of the prior
``register_checkpoint`` behaviour that overwrote a single role binding (one
specialist at a time). Each registered specialist becomes its own
``ModelDefinition`` whose ``model_id`` is the served adapter name and whose
``tags`` carry the routing key (``task:<sig>``, ``adapter:<id>``, ``base:<model>``),
so the existing tag-based ``pick_for_task`` routes to it with no registry schema
change.

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

    def __init__(self, registry: Any = None) -> None:
        self._by_task: dict[str, SpecialistAdapter] = {}
        self.registry = registry  # optional agent_utilities ModelRegistry

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
        """Add (or replace) the specialist for ``task_signature`` and register a model def."""
        spec = SpecialistAdapter(
            task_signature=task_signature,
            adapter_id=adapter_id,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
        )
        self._by_task[task_signature] = spec
        if self.registry is not None:
            from agent_utilities.models.model_registry import ModelDefinition

            # Append (do NOT overwrite a shared role): each specialist is its own
            # model, so N specialists on one base coexist and route by task tag.
            # Adapter identity rides in model_id (the served LoRA name) + tags, the
            # registry's sanctioned extension point — no schema change needed.
            self.registry.models = [m for m in self.registry.models if m.id != adapter_id]
            self.registry.models.append(
                ModelDefinition(
                    id=adapter_id,
                    name=f"SAI specialist: {task_signature}",
                    provider="vllm-lora",
                    model_id=adapter_id,
                    base_url=base_url,
                    tier=tier,  # type: ignore[arg-type]
                    tags=[
                        "specialist",
                        f"task:{task_signature}",
                        f"adapter:{adapter_id}",
                        f"base:{base_model}",
                    ],
                )
            )
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
            prompt, n=1, max_tokens=max_tokens, temperature=temperature, adapter=adapter_id
        )
        return out[0]["text"] if out else ""

    return generate
