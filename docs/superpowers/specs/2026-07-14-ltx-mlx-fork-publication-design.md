# LTX MLX fork publication design

## Goal

Publish the verified local LTX MLX stack under `raspverry` without pushing to the upstream repositories, and keep a simple path for merging future upstream ComfyUI updates.

## Repository ownership

- `comfyanonymous/ComfyUI` remains the ComfyUI upstream remote named `origin`.
- `raspverry/ComfyUI` is the maintained fork remote named `ours`.
- `dgrauet/ComfyUI-LTXVideo-mlx` remains the custom-node upstream remote named `origin`.
- `raspverry/ComfyUI-LTXVideo-mlx` is the maintained custom-node fork remote named `ours`.
- `dgrauet/ltx-2-mlx` is not forked. The current stack continues to use the verified `0.14.18` packages and custom-node compatibility path.

The custom-node fork is already published through commit `f0e6f3b05661e8a7e515e6f11bd74c8ed4fb688b`. The ComfyUI fork exists but does not receive the local stack until the documentation pin and verification below are complete.

## Publication

Update the operator guide to replace the temporary `local-only` warning with the maintained custom-node fork URL and full pinned commit. The guide will show an explicit clone and checkout path so another machine installs the tested revision rather than whichever commit happens to be on `master`.

The guide contract test will require the fork URL and full commit and will stop requiring `local-only`. No runtime network behavior, installer, dependency, or model format changes are added.

After tests pass, fast-forward local ComfyUI `master` to the completed hardening branch and push that commit to `ours/master`. Keep `origin` pointed at upstream and configure `ours` as the default push remote so an unqualified push cannot target upstream by accident. The custom-node repository uses the same remote policy.

## Future upstream updates

Use a merge-based update flow:

1. Fetch `origin`.
2. Merge `origin/master` into local `master` without rebasing or force-pushing published history.
3. Resolve conflicts in the smallest owning layer.
4. Run the full ComfyUI unit suite, LTX stack verifier, workflow tests, and relevant custom-node tests.
5. Push the verified merge only to `ours/master`.

If the custom-node fork changes, test it independently and update the pinned commit in the ComfyUI guide in the same verified change. Upstream pull requests may be prepared later, but they are outside this publication.

## Verification and failure handling

- The checked-in guide must name the maintained fork and exact custom-node commit.
- ComfyUI tests and the real local verifier must pass before publishing `ours/master`.
- Custom-node tests must pass at the pinned commit.
- After each push, GitHub's reported default-branch SHA must match the local SHA.
- A failed push changes no upstream state; keep the local branch and retry only after diagnosing the failure.
- Never force-push either maintained fork's `master` during normal synchronization.
