# Explicit Real-World Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real-world Gym registration explicit and dependency-isolated.

**Architecture:** `RealWorldEnv` imports the configured registration module
before `gym.make()`. The package root exposes existing names lazily, while
task modules keep standard Gym registration and existing ROS 1 behavior.

**Tech Stack:** Python, Gymnasium, Hydra/OmegaConf, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-explicit-realworld-registration-design.md`

## Global Constraints

- Base all work on `sync-main` (`c69eecbaa7eb6db06aa3318f103305703f3266c5`).
- Do not introduce F1 code, central ID mappings, proxy factories, or new setup flags.
- Preserve existing task-local `gym.register()` entry points and lazy public exports.
- Use test-first development and signed conventional commits.

---

### Task 1: Explicit registration loading and lazy package root

**Files:**
- Modify: `rlinf/envs/realworld/realworld_env.py`
- Modify: `rlinf/envs/realworld/__init__.py`
- Modify: existing ROS 1 task registration modules only as needed to retain
  their historical cleanup behavior
- Test: focused real-world registration unit tests

**Interfaces:**
- Consumes: `cfg.init_params.id` and `cfg.init_params.registration_module`
- Produces: selected task registration before `gym.make()` and lazy package exports

- [ ] Add failing tests for import ordering, invalid configuration, dependency
  isolation, lazy public exports, and scoped ROS 1 cleanup.
- [ ] Run the focused tests and record the expected failures.
- [ ] Implement the minimal explicit import and lazy exports without a central
  ID map or proxy registry.
- [ ] Run focused tests and Ruff.
- [ ] Commit the tested code.

### Task 2: Migrate shipped configs and documentation

**Files:**
- Modify: all shipped YAML files that select an existing real-world Gym ID
- Modify: relevant English and Chinese real-world configuration documentation
- Test: configuration composition and EN/ZH parity checks

**Interfaces:**
- Consumes: `init_params.registration_module` from Task 1
- Produces: runnable shipped configs with explicit task ownership

- [ ] Add failing coverage that enumerates shipped real-world configs and
  verifies their registration modules.
- [ ] Add the correct module to every affected YAML and document the field.
- [ ] Run focused config tests, Ruff, and relevant documentation checks.
- [ ] Commit the migration.

### Task 3: Integration verification and MR

**Files:**
- Modify only defects exposed by verification

- [ ] Run all relevant real-world unit/config tests and import-isolation probes.
- [ ] Review the complete diff against `sync-main` for unrelated changes.
- [ ] Push the branch to `glab` and create an MR targeting `sync-main`.
