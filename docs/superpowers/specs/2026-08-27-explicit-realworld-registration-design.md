# Explicit Real-World Registration Design

## Goal

Load only the selected real-world task stack before `gym.make()`, so one
robot does not require or initialize another robot's dependencies.

## Design

Every real-world environment config declares the module that owns its Gym
registrations:

```yaml
init_params:
  id: ButtonEnv-v1
  registration_module: rlinf.envs.realworld.xsquare.tasks
```

`RealWorldEnv` imports this module immediately before constructing the Gym
environment. Missing, empty, or invalid module names fail before
`gym.make()`.

`rlinf.envs.realworld` retains its documented public exports through lazy
attribute imports, but importing the package no longer imports every robot
stack or registers every task.

Existing ROS 1 cleanup behavior remains owned by the existing ROS 1 task
stacks. Loading an existing ROS 1 registration module may perform that
historical cleanup; loading an unrelated or future ROS 2 module must not.
This change neither generalizes nor removes the cleanup behavior.

Direct Gym users must explicitly import the selected task module before
calling `gym.make()`. Importing only `rlinf.envs.realworld` no longer
registers all task IDs.

## Scope

- Add `init_params.registration_module` to every existing real-world YAML.
- Import that module in `RealWorldEnv` before `gym.make()`.
- Make the `realworld` package root dependency-isolated and lazy.
- Preserve standard task-local `gym.register()` calls.
- Preserve existing public package exports lazily.
- Do not add a central environment-ID registry, proxy factories, or F1 code.
- Do not change Gymnasium compatibility, replay, runner, or actor behavior.

## Verification

- Each shipped real-world config identifies the correct task module.
- The requested module is imported before `gym.make()`.
- Invalid registration modules fail clearly before environment construction.
- Importing the package root does not load sibling robot stacks or trigger
  ROS cleanup.
- Explicitly loading one task module registers its IDs without loading the
  other robot stacks.
- Existing public exports remain available through lazy attribute access.

