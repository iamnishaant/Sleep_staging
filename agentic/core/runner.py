"""The state machine runner.

A workflow is an ordered set of named steps. Each step function receives a StepContext, writes its
outputs into its own directory, and returns a Decision naming the next step (or None to finish).
The runner:
  * computes the run ID from the input, the policy and the release (execution manifest);
  * replays finished steps from their DONE markers (never re-runs them);
  * wipes and re-runs a step that started but did not finish;
  * traces every transition;
  * bounds the walk, so a run always ends.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import atomic
from .crash import crash_point
from .hashing import sha_text
from .policy import Policy
from .release import Release
from .store import RunStore
from .trace import Trace


def make_run_id(input_sha256: str, policy_sha256: str, manifest_sha256: str) -> str:
    return sha_text(f"{input_sha256}|{policy_sha256}|{manifest_sha256}")[:24]


@dataclass
class Decision:
    next: str | None                      # name of the next step, or None = finished
    reasons: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


@dataclass
class StepContext:
    run_id: str
    step_dir: Path
    run_dir: Path
    store: RunStore
    release: Release
    policy: Policy
    services: dict                        # injected collaborators (report backend, ...)
    trace: Trace

    def out(self, name: str) -> Path:
        return self.step_dir / name

    def earlier(self, step_name: str) -> Path:
        """Directory of an earlier, finished step of this run."""
        hits = sorted(self.run_dir.glob(f"*_{step_name}"))
        if not hits:
            raise KeyError(f"step {step_name!r} has no outputs in this run")
        return hits[-1]


Step = Callable[[StepContext], Decision]


@dataclass
class Workflow:
    name: str
    first: str
    steps: dict[str, Step]
    max_transitions: int = 30


class Runner:
    def __init__(self, workflow: Workflow, release: Release, policy: Policy, store: RunStore,
                 services: dict | None = None):
        self.workflow, self.release, self.policy, self.store = workflow, release, policy, store
        self.services = services or {}

    def run(self, input_sha256: str, seed_files: dict[str, bytes]) -> dict:
        """Run (or resume) the workflow for one input. `seed_files` are written into the run directory
        before the first step (e.g. the packet). Returns the delivery record."""
        self.release.check_pins()
        manifest_sha = self.release.manifest_sha256
        run_id = make_run_id(input_sha256, self.policy.sha256, manifest_sha)
        run_dir = self.store.run_dir(run_id)
        trace = Trace(self.store.trace_path, run_id, self.policy.sha256, manifest_sha)

        if (rec := self.store.delivered(run_id)) is not None:
            trace.log("-", "already_delivered")
            return rec

        if not (run_dir / "run.json").exists():
            for name, data in seed_files.items():
                atomic.write_bytes(run_dir / "input" / name, data)
            atomic.write_json(run_dir / "run.json", {
                "run_id": run_id, "workflow": self.workflow.name, "input_sha256": input_sha256,
                "policy_sha256": self.policy.sha256, "policy_version": self.policy.version,
                "release": self.release.name, "execution_manifest_sha256": manifest_sha,
                "execution_manifest": self.release.manifest()["core"]})
            trace.log("-", "run_created", workflow=self.workflow.name, release=self.release.name)
        else:
            trace.log("-", "run_resumed")

        step_name, index = self.workflow.first, 1
        for _ in range(self.workflow.max_transitions):
            if step_name is None:
                rec = self.store.delivered(run_id)
                if rec is None:
                    raise RuntimeError(f"run {run_id}: workflow ended without a delivery")
                return rec
            step_dir = self.store.step_dir(run_id, index, step_name)
            done = self.store.done(step_dir)
            if done is not None:                                  # finished earlier: replay its decision
                step_name, index = done["next"], index + 1
                continue
            self.store.reset(step_dir)
            ctx = StepContext(run_id, step_dir, run_dir, self.store, self.release, self.policy,
                              self.services, trace)
            trace.log(step_name, "start", index=index)
            t0 = time.monotonic()
            decision = self.workflow.steps[step_name](ctx)
            crash_point(f"{step_name}:before_done")
            self.store.mark_done(step_dir, {"step": step_name, "next": decision.next,
                                            "reasons": decision.reasons, "info": decision.info})
            trace.log(step_name, "done", next=decision.next, reasons=decision.reasons, info=decision.info,
                      wall_s=round(time.monotonic() - t0, 3))
            crash_point(f"{step_name}:after_done")
            step_name, index = decision.next, index + 1
        raise RuntimeError(f"run {run_id}: exceeded {self.workflow.max_transitions} transitions")
