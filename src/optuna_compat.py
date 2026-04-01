from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

try:
    import optuna as _real_optuna
    from optuna.pruners import MedianPruner  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore
    from optuna.study import Study  # type: ignore
    from optuna.trial import FrozenTrial  # type: ignore

    optuna = _real_optuna
    USING_FAKE_OPTUNA = False
except ModuleNotFoundError:
    import math
    import random

    USING_FAKE_OPTUNA = True

    class TrialState(Enum):
        COMPLETE = "COMPLETE"
        PRUNED = "PRUNED"
        FAIL = "FAIL"

    class TrialPruned(Exception):
        pass

    @dataclass
    class FrozenTrial:
        number: int
        value: float | None
        params: Dict[str, Any]
        user_attrs: Dict[str, Any]
        state: TrialState

    class Trial:
        def __init__(self, number: int, sampler_seed: int = 42) -> None:
            self.number = number
            self.params: Dict[str, Any] = {}
            self.user_attrs: Dict[str, Any] = {}
            self._reported_steps: Dict[int, float] = {}
            self._rng = random.Random((sampler_seed + 1) * 100000 + number)
            self.value: float | None = None
            self.state = TrialState.COMPLETE

        def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
            if name in self.params:
                return self.params[name]
            value = choices[self._rng.randrange(len(choices))]
            self.params[name] = value
            return value

        def suggest_int(self, name: str, low: int, high: int) -> int:
            if name in self.params:
                return int(self.params[name])
            value = self._rng.randint(low, high)
            self.params[name] = value
            return value

        def suggest_float(self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False) -> float:
            if name in self.params:
                return float(self.params[name])
            if log:
                value = math.exp(self._rng.uniform(math.log(low), math.log(high)))
            elif step is not None and step > 0:
                count = int(round((high - low) / step))
                value = low + step * self._rng.randint(0, count)
            else:
                value = self._rng.uniform(low, high)
            self.params[name] = value
            return value

        def report(self, value: float, step: int) -> None:
            self._reported_steps[int(step)] = float(value)

        def should_prune(self) -> bool:
            return False

        def set_user_attr(self, key: str, value: Any) -> None:
            self.user_attrs[key] = value

    class MedianPruner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class TPESampler:
        def __init__(self, seed: int = 42, *args: Any, **kwargs: Any) -> None:
            self.seed = seed
            self.args = args
            self.kwargs = kwargs

    @dataclass
    class Study:
        study_name: str
        direction: str
        sampler: TPESampler | None = None
        pruner: MedianPruner | None = None
        storage: str | None = None
        trials: List[FrozenTrial] = field(default_factory=list)

        def optimize(
            self,
            objective: Callable[[Trial], float],
            n_trials: int,
            gc_after_trial: bool = True,
            show_progress_bar: bool = False,
            callbacks: List[Callable[["Study", FrozenTrial], None]] | None = None,
        ) -> None:
            callbacks = callbacks or []
            sampler_seed = 42 if self.sampler is None else int(self.sampler.seed)
            start_number = len(self.trials)
            for offset in range(int(n_trials)):
                trial = Trial(number=start_number + offset, sampler_seed=sampler_seed)
                try:
                    value = objective(trial)
                    frozen = FrozenTrial(
                        number=trial.number,
                        value=float(value) if value is not None else None,
                        params=dict(trial.params),
                        user_attrs=dict(trial.user_attrs),
                        state=TrialState.COMPLETE,
                    )
                except TrialPruned:
                    frozen = FrozenTrial(
                        number=trial.number,
                        value=None,
                        params=dict(trial.params),
                        user_attrs=dict(trial.user_attrs),
                        state=TrialState.PRUNED,
                    )
                self.trials.append(frozen)
                for callback in callbacks:
                    callback(self, frozen)

    def create_study(
        *,
        study_name: str,
        direction: str,
        sampler: TPESampler | None = None,
        pruner: MedianPruner | None = None,
        storage: str | None = None,
        load_if_exists: bool = True,
    ) -> Study:
        return Study(study_name=study_name, direction=direction, sampler=sampler, pruner=pruner, storage=storage)

    optuna = SimpleNamespace(
        Trial=Trial,
        TrialPruned=TrialPruned,
        create_study=create_study,
    )
