# Handoff: context outside the design doc

For the next agent picking up "where the llama keeps its grammar". Read the
design first:
`docs/superpowers/specs/2026-10-26-where-the-llama-keeps-its-grammar-design.md`.
This file holds what the design does not: status, why things were decided,
what was ruled out, and practical notes about the machine and repos.

## Status (as of 2026-10-28)

- Brainstorming is done. The design doc is written, reviewed by a builder
  check, and pushed to `main` of
  https://github.com/jasondew/interpretability_talk.
- **The speaker has not yet explicitly approved the design.** Get that
  approval before building anything. Once approved, the next step is the
  implementation plan via `superpowers:writing-plans`, and nothing else.
- No code exists yet in either repo for this talk. This repo holds only
  docs; it is not yet a Phoenix app.
- Talk date: Tuesday, December 1, 2026. Weekly checkpoints are in the
  design's "Open risks" section. Treat them as real; the speaker agreed to
  them as a guard against scope creep.

## The speaker

- Jason, gives talks to the AI Guild, which meets every Tuesday. The Guild
  saw the first talk, "the llama who chases the dogs".
- Wants a direct, unflattering advisor: push back on weak reasoning, call
  out scope creep and wasted time. Their global CLAUDE.md says so.
- **Stated top worry: not being able to explain the topic or answer
  questions.** Everything about build order (toy model first), the hard
  questions doc, and running an experiment with an unknown outcome exists
  to address this. When building, explain results to the speaker as you
  go; do not just produce numbers. Ask them to predict an outcome before
  revealing it when that helps them own it.
- Was fine dropping Elixir purity "unless it's an easy win", but
  specifically asked for Popcorn, so the Elixir in the browser angle
  matters to them.
- On imperfect results: "the value is in the journey." Honest framing
  beats clean looking results.

## How the idea evolved (so you do not reopen settled questions)

The speaker started with: add layers, move to mixture of experts, run in
the browser via WASM, about interpretability. What changed and why:

- **MoE dropped.** It does not serve interpretability unless the router is
  the subject, and it makes the model harder to explain.
- **More layers dropped.** "Towards Monosemanticity" used one layer; more
  depth adds things to explain and shows nothing new.
- **"Zoom into a neuron and find a clean concept" reframed.** Raw neurons in
  small models are usually polysemantic because of superposition. The
  talk's arc now starts with the messy neuron and gets to clean features
  through an SAE. The speaker loved this.
- **Tokenizer and bigger vocabulary for its own sake rejected.** The
  speaker asked about this directly. The answer: the lever is more
  *concepts* than dimensions, which comes from extending the grammar and
  shrinking `d_model`. Vocabulary grows as a side effect. One word per
  token keeps features readable.
- **The grammar is the core asset.** Known ground truth lets features be
  scored with precision and recall instead of eyeballed. Protect this.

## Title

Chosen: "where the llama keeps its grammar". The speaker liked the "what"
and "where" veins. Runners up, in case the title is revisited:
"what the llama knows", "where the llama keeps its plurals", "how the llama
knows it's plural", "the dogs who chase the llama". Subtitle is a working
one.

## Repos and layout

All under `~/src/talks/`:

- `tiny_llm/`: the first talk's model, pure Elixir stdlib, zero deps.
  Read its `CLAUDE.md` and `docs/build-brief.md` before touching it. Its
  hard constraints (zero deps, hand written backprop, fixed vocab and
  grammar) still bind the original code. New work goes in a nested Mix
  project at `tiny_llm/interpretability/` with its own `CLAUDE.md`
  stating which constraints are relaxed. Do not modify existing files
  there beyond adding a short pointer to the README and `CLAUDE.md`.
- `tiny_llm_talk/`: the first talk's LiveView deck. Use it as the template
  for `interpretability_talk`. Worth reading before planning the deck:
  - `mix.exs`: path dep on `../tiny_llm`, Phoenix 1.8, LiveView 1.2,
    Tailwind with daisyUI, esbuild, makeup for code slides.
  - `lib/tiny_llm_talk/{deck,slide,section}.ex`: how slides are modeled.
  - `lib/tiny_llm_talk/{checkpoint,trainer}.ex` and
    `lib/mix/tasks/talk.train.ex`: how checkpoints are produced and
    checked (to 1.0e-9; this talk relaxes that, see the design).
  - `lib/tiny_llm_talk_web/live/{deck_live,presenter_live}.ex`: audience
    view and presenter view.
  - `Dockerfile`, `fly.toml`: deployment pattern the explorer will reuse.
  - `docs/talk-outline.md`: the outline format to copy for this talk.
- `interpretability_talk/`: this repo. Will become the LiveView deck with
  a path dep on `../tiny_llm/interpretability`.

## Machine notes

- macOS, zsh with fish as login shell. Use macOS compatible commands.
- System Elixir is 1.18.4 on OTP 28. `tiny_llm_talk/mise.toml` pins Erlang
  27.3.4.11 and Elixir 1.18.4-otp-27; run the deck through `mise exec`.
  Check whether EXLA builds cleanly on whichever OTP you pick before
  committing to it, and pin the new projects with `mise.toml` too.
- `tiny_llm` has a stale, locked worktree entry pointing at
  `/Users/jasondew/src/tiny_llm/.claude/worktrees/jdew+talk_outline`, a
  path that no longer exists (the repo moved under `talks/`). Harmless;
  ask the speaker before pruning it.
- Creating a public GitHub repo was blocked by the permission classifier;
  the speaker ran `gh repo create` themselves. Expect the same for any new
  public surface (Fly deploys included). Hand the command to the speaker.

## Git workflow (from the speaker's global rules)

- Branches are `jdew/<descriptive_name_with_underscores>`.
- Work in a git worktree off `main`, push and open a PR from there, watch
  CI, address review. The initial design commit went straight to `main`
  only because the repo was empty.
- Commit messages explain why. No em dashes or dashes in prose anywhere.

## Popcorn research notes

From Software Mansion's write up and docs (checked 2026-10-28):

- Runs Elixir in the browser by compiling AtomVM (a small Erlang VM for
  microcontrollers) to WASM. The full BEAM on WASM was not feasible.
- v1.0.0 on hexdocs (`https://popcorn.hexdocs.pm/`, redirected from
  `hexdocs.pm/popcorn`). The docs page body did not render through the
  fetch tool; read it in a browser or pull the package source.
- AtomVM supports a subset of the standard library. Verify `:math`,
  float list operations, and whatever `TinyLlm.Tensor` uses before
  assuming reuse works.
- Bundle: AtomVM about 190KB compressed, over 3MB with Erlang and Elixir
  libraries.
- Code runs in an isolated iframe with JS interop.
- Software Mansion calls it unsuitable for production. Fine for a demo,
  but do the two day time box early (week of November 23 at the latest,
  ideally a spike sooner) so a failure leaves time for the JS fallback.
- Nx does not run on AtomVM. Hence the plain Elixir forward pass in the
  browser and the parity test against the Nx model.

## Reference papers

- Anthropic, "Toy Models of Superposition" (2022): the toy model for
  section 3 of the arc.
- Anthropic, "Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning" (2023): one layer transformer, SAE on MLP
  activations, feature splitting with dictionary size. The closest prior
  art to this talk; read it before planning the SAE.
