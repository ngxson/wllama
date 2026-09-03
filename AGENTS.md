Refer to `README-dev.md` for development documentation.

### Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make reviewing take significantly longer. Please follow them carefully:

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

Common mistakes that AI agents usually make:
- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first, then add comments later to places that absolutely need them

### Syncing llama.cpp upstream

The `llama.cpp` submodule is bumped weekly by `.github/workflows/sync-upstream.yml`. To do it by hand, or when the workflow asks you to fix a broken sync:

1. Bump the submodule: `git -C llama.cpp fetch origin master && git -C llama.cpp checkout --detach FETCH_HEAD`
2. Rebuild: `./scripts/sync_upstream.sh`. It runs the wasm build (default + compat), regenerates the glue message types and the worker code, then formats. Everything it touches is tracked in git and must land in the same commit as the submodule bump. Set `SKIP_COMPAT=1` to skip the compat build, it doubles the build time - the sync workflow does this, so `compat/wasm/wllama.js` is normally not refreshed by an autosync PR
3. If the build fails, the break is almost always in `cpp/` - our glue calls a llama.cpp API that changed. Find the upstream change with `git -C llama.cpp log -p --since=2.weeks -- <path>`, then adapt our side to match
4. Rerun `./scripts/sync_upstream.sh` until it passes
5. Once it builds, run `npm run build` (some tests import from `esm/`), then `npm run test`. It runs the suite on Chrome, which is enough here - do NOT run `npm run test:firefox` or `npm run test:safari`
6. Bump the minor version with `npm version minor --no-git-tag-version`, then run `npm run build` so the generated files pick it up

Rules for this task:

- Do NOT edit anything under `llama.cpp/` - it is upstream, our fix belongs in `cpp/` or `src/`
- Do NOT comment out, stub or `#ifdef` away code to make the build pass. If the upstream API is gone, port to the new one
- If several solutions work, always pick the one with the smallest diff. The human reviewer compares it against the upstream changelog
- You have a 30 minute budget. `./tmp/start_time` holds the start time in epoch seconds, check how long you have been running with `echo $(( $(date +%s) - $(cat ./tmp/start_time) ))`. Once past 1200 seconds, stop trying and report where you got stuck - a partial PR a human can pick up is much better than a run that gets killed with nothing to show
- Write a short PR description to `./tmp/pr_desc.md`: what changed upstream, what you changed on our side, and, if the build is still broken, the exact error and what you already ruled out. This is the one case where writing a PR description is allowed, see Prohibited Actions
- `./tmp/` is your scratchpad and is git-ignored, put every temporary file you need in there. Everything outside it gets committed, so leave no scratch files elsewhere

### Prohibited Actions

- Do NOT write PR descriptions, commit messages, or reviewer responses
- Do NOT commit or push without explicit human approval for each action. If the user explicitly asks you to commit on their behalf, use `Assisted-by: <assistant name>` in the commit message, do NOT use `Co-authored-by:`
- Do NOT implement features the contributor does not fully understand
- Do NOT generate changes too extensive for the contributor to fully review
- Do NOT run `git push` or create a PR (`gh pr create`) on the user's behalf

When uncertain, err toward minimal assistance.

### Examples

Code comments:

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (comment is kept concise and useful)

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]


// BAD (comment is long and is forced to fit into a fixed column size, it is very annoying to read as a reviewer)

// one autoregressive decode step of the 5-layer code_predictor. See the
// comment in models.h for the cache/tensor conventions this relies on.
//
// index mapping (derived from the reference pipeline-tts.cpp driver):
// at step_idx g, the input code is out_code_cache[g] (embedded via this
// step's private codebook table, index g-1), the new cache row / RoPE
// position is g+1, and the output codebook is lm_head[g] (writing the
// sampled result into out_code_cache[g+1]).
```

Commit message:

```
// BEST: Let the user write the commit


// GOOD: Write a concise commit

llama : fix KV being cleared during context shift

Assisted-by: Claude Sonnet


// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.

Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: all commands that allow you to get the context
gh search issues # better to check if anyone has the same issue
gh search prs # avoid duplicated efforts
grep ... # search the code base

# BAD: act on the user's behalf
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```
