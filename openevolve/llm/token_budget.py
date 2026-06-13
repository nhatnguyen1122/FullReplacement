"""
Token usage tracking and run-level token budget enforcement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


class TokenBudgetExceeded(RuntimeError):
    """Raised when the configured run-level token budget has been exceeded."""


@dataclass
class TokenUsage:
    """Cumulative token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0


class TokenBudget:
    """Tracks token usage, optionally backed by multiprocessing shared state."""

    def __init__(
        self,
        max_total_tokens: Optional[int] = None,
        shared_state: Optional[Any] = None,
        lock: Optional[Any] = None,
    ):
        self.max_total_tokens = max_total_tokens
        self._local = TokenUsage()
        self._shared_state = shared_state
        self._lock = lock

        if self._shared_state is not None:
            self._ensure_shared_state()

    def _ensure_shared_state(self) -> None:
        for key, value in asdict(TokenUsage()).items():
            if key not in self._shared_state:
                self._shared_state[key] = value

    def reset(
        self,
        max_total_tokens: Optional[int] = None,
        shared_state: Optional[Any] = None,
        lock: Optional[Any] = None,
    ) -> None:
        self.max_total_tokens = max_total_tokens
        self._local = TokenUsage()
        self._shared_state = shared_state
        self._lock = lock
        if self._shared_state is not None:
            self._ensure_shared_state()
            self._with_lock(self._reset_shared)

    def _reset_shared(self) -> None:
        for key, value in asdict(TokenUsage()).items():
            self._set_shared_int(key, value)

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> TokenUsage:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        total_tokens = int(total_tokens or 0)

        if total_tokens <= 0 and prompt_tokens <= 0 and completion_tokens <= 0:
            return self.snapshot()

        if self._shared_state is not None:
            usage = self._with_lock(
                lambda: self._record_shared(prompt_tokens, completion_tokens, total_tokens)
            )
        else:
            self._local.prompt_tokens += prompt_tokens
            self._local.completion_tokens += completion_tokens
            self._local.total_tokens += total_tokens
            self._local.requests += 1
            usage = self.snapshot()

        self._raise_if_exceeded(usage)
        return usage

    def _record_shared(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> TokenUsage:
        self._ensure_shared_state()
        self._set_shared_int("prompt_tokens", self._get_shared_int("prompt_tokens") + prompt_tokens)
        self._set_shared_int(
            "completion_tokens", self._get_shared_int("completion_tokens") + completion_tokens
        )
        self._set_shared_int("total_tokens", self._get_shared_int("total_tokens") + total_tokens)
        self._set_shared_int("requests", self._get_shared_int("requests") + 1)
        return self.snapshot()

    def snapshot(self) -> TokenUsage:
        if self._shared_state is None:
            return TokenUsage(**asdict(self._local))

        self._ensure_shared_state()
        return TokenUsage(
            prompt_tokens=self._get_shared_int("prompt_tokens"),
            completion_tokens=self._get_shared_int("completion_tokens"),
            total_tokens=self._get_shared_int("total_tokens"),
            requests=self._get_shared_int("requests"),
        )

    def _get_shared_int(self, key: str) -> int:
        value = self._shared_state.get(key, 0)
        if hasattr(value, "value"):
            return int(value.value)
        return int(value or 0)

    def _set_shared_int(self, key: str, value: int) -> None:
        current = self._shared_state.get(key)
        if hasattr(current, "value"):
            current.value = int(value)
        else:
            self._shared_state[key] = int(value)

    def exceeded(self) -> bool:
        return (
            self.max_total_tokens is not None
            and self.max_total_tokens >= 0
            and self.snapshot().total_tokens > self.max_total_tokens
        )

    def _raise_if_exceeded(self, usage: TokenUsage) -> None:
        if (
            self.max_total_tokens is not None
            and self.max_total_tokens >= 0
            and usage.total_tokens > self.max_total_tokens
        ):
            raise TokenBudgetExceeded(
                "Token budget exceeded: "
                f"{usage.total_tokens} total tokens used "
                f"(limit {self.max_total_tokens})"
            )

    def _with_lock(self, fn):
        if self._lock is None:
            return fn()
        with self._lock:
            return fn()


_GLOBAL_TOKEN_BUDGET = TokenBudget()


def configure_token_budget(
    max_total_tokens: Optional[int] = None,
    shared_state: Optional[Any] = None,
    lock: Optional[Any] = None,
) -> TokenBudget:
    """Configure the process-global token budget tracker."""

    _GLOBAL_TOKEN_BUDGET.reset(
        max_total_tokens=max_total_tokens,
        shared_state=shared_state,
        lock=lock,
    )
    return _GLOBAL_TOKEN_BUDGET


def record_token_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: Optional[int] = None,
) -> TokenUsage:
    """Record token usage in the process-global tracker."""

    return _GLOBAL_TOKEN_BUDGET.record(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def get_token_usage() -> TokenUsage:
    """Return a snapshot of the process-global token usage."""

    return _GLOBAL_TOKEN_BUDGET.snapshot()


def is_token_budget_exceeded() -> bool:
    """Return whether the process-global token budget is exceeded."""

    return _GLOBAL_TOKEN_BUDGET.exceeded()


def token_usage_to_dict(usage: Optional[TokenUsage] = None) -> Dict[str, int]:
    """Serialize token usage for logs, checkpoints, or worker results."""

    return asdict(usage or get_token_usage())
