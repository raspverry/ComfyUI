# LTX MLX Fork Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the verified LTX MLX stack on the `raspverry` forks, pin the custom-node fork revision, and make future upstream synchronization safe and repeatable.

**Architecture:** Keep `origin` as the upstream fetch remote and `ours` as the only normal push target in both repositories. Replace the temporary local-only documentation contract with an exact maintained-fork URL and commit, then fast-forward local ComfyUI `master` and publish it directly to `ours/master` after full verification.

**Tech Stack:** Git, GitHub CLI, Python, pytest, Markdown

---

### Task 1: Pin the maintained custom-node fork

**Files:**
- Modify: `tests-unit/ltx_stack/test_model_manifest.py:467`
- Modify: `docs/ltx-mlx-stack.md:14`

- [ ] **Step 1: Replace the local-only guide contract with a fork pin contract**

Rename the guide test and require the published fork URL and full commit:

```python
def test_operator_guide_records_runtime_requirements():
    guide = (ROOT / "docs/ltx-mlx-stack.md").read_text()

    for requirement in (
        "64 GB",
        "reference-front.png",
        "reference-profile.png",
        "https://github.com/raspverry/ComfyUI-LTXVideo-mlx",
        "f0e6f3b05661e8a7e515e6f11bd74c8ed4fb688b",
        "HF_HUB_OFFLINE=1",
    ):
        assert requirement in guide
    assert "local-only" not in guide
```

- [ ] **Step 2: Run the guide contract and verify RED**

Run:

```bash
/Users/hansol/dev/oss/comfy/.venv/bin/python -m pytest -q \
  tests-unit/ltx_stack/test_model_manifest.py::test_operator_guide_records_runtime_requirements
```

Expected: fail because the guide still says `local-only` and does not contain the maintained fork URL or full commit.

- [ ] **Step 3: Replace the temporary warning with pinned installation instructions**

Replace the paragraph at `docs/ltx-mlx-stack.md:14` with concise maintained-fork guidance and commands:

````markdown
The maintained custom node is [raspverry/ComfyUI-LTXVideo-mlx](https://github.com/raspverry/ComfyUI-LTXVideo-mlx), pinned here to commit `f0e6f3b05661e8a7e515e6f11bd74c8ed4fb688b`. Stock `dgrauet/ComfyUI-LTXVideo-mlx` does not contain the Ingredients node or the in-process I2V fix used by these workflows.

```bash
git clone https://github.com/raspverry/ComfyUI-LTXVideo-mlx.git custom_nodes/ComfyUI-LTXVideo-mlx
git -C custom_nodes/ComfyUI-LTXVideo-mlx checkout f0e6f3b05661e8a7e515e6f11bd74c8ed4fb688b
```
````

- [ ] **Step 4: Run focused and full guide tests**

Run:

```bash
/Users/hansol/dev/oss/comfy/.venv/bin/python -m pytest -q \
  tests-unit/ltx_stack/test_model_manifest.py::test_operator_guide_records_runtime_requirements
/Users/hansol/dev/oss/comfy/.venv/bin/python -m pytest -q tests-unit/ltx_stack/test_model_manifest.py
git diff --check
```

Expected: focused test passes, the manifest suite passes, and the diff check exits 0.

- [ ] **Step 5: Commit the fork pin**

```bash
git add docs/ltx-mlx-stack.md tests-unit/ltx_stack/test_model_manifest.py
git commit -m "Pin LTX MLX custom node fork"
```

### Task 2: Verify and publish the maintained ComfyUI fork

**Files:**
- Modify local Git configuration only in `/Users/hansol/dev/oss/comfy`
- Modify local Git configuration only in `/Users/hansol/dev/oss/comfy/custom_nodes/ComfyUI-LTXVideo-mlx`
- Update external branch: `raspverry/ComfyUI:master`

- [ ] **Step 1: Verify repository scope and fork identity**

Run:

```bash
git status --short --untracked-files=no
git -C /Users/hansol/dev/oss/comfy status --short --untracked-files=no
git -C /Users/hansol/dev/oss/comfy/custom_nodes/ComfyUI-LTXVideo-mlx status --short
gh repo view raspverry/ComfyUI --json isFork,parent,defaultBranchRef
gh repo view raspverry/ComfyUI-LTXVideo-mlx --json isFork,parent,defaultBranchRef
```

Expected: the hardening, local `master`, and custom-node tracked worktrees are clean; both GitHub repositories are forks with default branch `master` and the intended upstream parent.

- [ ] **Step 2: Make `ours` the default push remote**

Run:

```bash
git config remote.pushDefault ours
git -C /Users/hansol/dev/oss/comfy/custom_nodes/ComfyUI-LTXVideo-mlx config remote.pushDefault ours
git config --get remote.pushDefault
git -C /Users/hansol/dev/oss/comfy/custom_nodes/ComfyUI-LTXVideo-mlx config --get remote.pushDefault
```

Expected: both configuration reads print `ours`. Keep `origin` URLs unchanged so future `git fetch origin` reads upstream.

- [ ] **Step 3: Run fresh pre-publish verification**

From the hardening worktree, run:

```bash
/Users/hansol/dev/oss/comfy/.venv/bin/python -m pytest -q tests-unit
/Users/hansol/dev/oss/comfy/.venv/bin/python scripts/ltx_stack/verify_install.py --root .
bash -n scripts/start_ltx_stack_macos.sh
git diff 52f894ea..HEAD --check
```

From the custom-node checkout, run:

```bash
/Users/hansol/dev/oss/comfy/.venv/bin/python -m pytest -q tests
```

Expected: ComfyUI has no failures, the verifier exits 0 with Ingredients optional, shell syntax and diff checks exit 0, and the custom node reports 36 passing tests.

- [ ] **Step 4: Fast-forward local ComfyUI master**

Run from `/Users/hansol/dev/oss/comfy`:

```bash
git merge --ff-only codex/ltx-mlx-hardening
git rev-parse master
git rev-parse codex/ltx-mlx-hardening
```

Expected: the merge is a fast-forward and both SHA outputs match. Preserve the unrelated untracked `.serena/` directory.

- [ ] **Step 5: Publish only the maintained fork**

Run:

```bash
git push ours master:master
```

Expected: `raspverry/ComfyUI:master` advances to local `master`. Do not push `origin` and do not create an upstream pull request.

- [ ] **Step 6: Verify both published default branches**

Run:

```bash
gh api repos/raspverry/ComfyUI/commits/master --jq .sha
git rev-parse master
gh api repos/raspverry/ComfyUI-LTXVideo-mlx/commits/master --jq .sha
git -C /Users/hansol/dev/oss/comfy/custom_nodes/ComfyUI-LTXVideo-mlx rev-parse HEAD
git status --short --branch
git -C /Users/hansol/dev/oss/comfy/custom_nodes/ComfyUI-LTXVideo-mlx status --short --branch
```

Expected: each remote SHA matches its local SHA. ComfyUI may still report the pre-existing untracked `.serena/`; there must be no tracked changes. The custom-node checkout must remain clean.
