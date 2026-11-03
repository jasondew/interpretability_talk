# Browser Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static page where someone types or picks a sentence and sees, word by word, what the llama model predicts next, which SAE features (or raw neurons) fire, what each one is about, and what it fired on most strongly in the evaluation sentences. The page runs the model's forward pass in plain JS, and a test proves it matches Nx.

**Architecture:** A mix task, `interp.export`, writes the model, the 512 feature SAE, per unit scores, top examples and steering directions to `interpretability/explorer/data/`. It also writes parity fixtures, computed by Nx, to `interpretability/explorer/test/`. The page is plain ES modules, with no build step and no dependencies. The math (`model.js`, `sae.js`, `sentence.js`) is pure and tested with `node --test` against the fixtures. A thin DOM layer, `main.js`, draws the page and is checked by hand in a browser at phone width.

**Tech Stack:** Elixir 1.18 (built in `JSON` module), Nx and EXLA for the export, plain JS (ES modules, `Float32Array`), and Node 23's built in test runner.

**Spec:** `docs/superpowers/specs/2026-10-26-where-the-llama-keeps-its-grammar-design.md`, section "Browser explorer" and "Hosting the explorer".

## Global Constraints

- "The baseline is plain JS reading the exported weights."
- "A test must check that the JS forward pass matches the Nx forward pass on the same weights, to a stated tolerance." Stated tolerance: **1.0e-4 absolute** on logits and neurons. The SAE's active features must be the same set, except where two candidates are within 1.0e-4 of the cutoff.
- "Steering in the browser is a nice to have." It is the last task and is cut first.
- Popcorn is out of scope for this plan: "attempted only after the baseline works", in a separate plan, time boxed to about two days.
- Acceptance includes a phone over mobile data "loading it in under about ten seconds and then responding to a click within a second". This plan can't test that on a real phone, because hosting belongs to the deck plan. It sets budgets that make it likely: **data under 1.5 MB gzipped**, and **one forward pass under 50 ms** in Node.
- Tiny_llm's rules apply: never change the parent project's files, keep the model size fixed, and keep the checkpoints intact. The export only reads `model.etf` and `sae_512.etf`.
- Elixir style: no leading pipe, `## PRIVATE FUNCTIONS`, no abbreviations, no single letter names. In JS, use the same naming rule (`activation`, not `act`).
- Prose (UI copy, docs): no dashes used as punctuation.

## Rulings made while planning (flagged for the speaker)

- **Where it lives:** `tiny_llm/interpretability/explorer/`, not the deck repo. The parity test needs Nx and the JS side by side, and the deck app doesn't exist yet. The deck plan will copy `explorer/` into its `priv/static/explorer/`. Cost if wrong: one directory move.
- **Branch:** `explorer`, off `end_to_end_sae` (PR #6), so it's stacked on #3 through #6. Cost if wrong: a rebase.
- **Weights format:** one `weights.bin` of little endian float32, plus a JSON manifest of names, shapes and offsets. It's exact (no decimal rounding to break parity) and about 600 KB. JSON numbers would be about 1.8 MB.
- **Neurons and features both:** a toggle shows raw neurons or SAE features, because the head to head comparison is the talk's thesis. The extra data is small (128 units).
- **Which SAE:** the 512 feature reconstruction SAE (`sae_512.etf`), the one the head to head results are about.
- **Node in mise:** add `node = "23.11.0"` to `interpretability/mise.toml`, so `node --test` is reproducible.

## Review Focus

1. **Words outside the vocabulary, capital letters, punctuation, extra spaces.** The page lowercases the text and strips `.,!?` before checking. Unknown words are named in a message, and nothing crashes.
2. **Sentences longer than the context.** At most 19 words fit after the start token. Longer input gets a message, not a silent cut.
3. **Ties at the TopK cutoff.** Nx keeps every feature whose value is `>=` the 8th largest, which can be more than 8. JS must use the same rule, not "exactly 8".
4. **Dead units.** Of the 512 features, 31 never fire. Clicking one must show "never fires on the evaluation sentences", not an empty or broken panel.
5. **Phone width.** At 375 px the token row wraps, tap targets are at least 40 px, and there's no horizontal scroll.

Each of these gets its test in the task that owns the code: 1 and 2 in Task 3, 3 in Task 2, 4 in Task 1 (export) and Task 4 (manual check), and 5 in Task 4 (manual check).

---

### Task 1: Export the model, SAE and unit data

**Files:**
- Create: `interpretability/lib/interpretability/export.ex`
- Create: `interpretability/lib/mix/tasks/interp.export.ex`
- Create: `interpretability/test/interpretability/export_test.exs`
- Modify: `interpretability/.gitignore`. Nothing to ignore: the exported data is committed, because the page is static and the deck copies it.
- Modify: `interpretability/mise.toml` (add node)

**Interfaces:**
- Produces:
  - `Export.weights(%{atom() => Nx.Tensor.t()}) :: {manifest :: map(), binary()}`. The manifest is `%{"tensors" => [%{"name" => String.t(), "shape" => [pos_integer()], "offset" => non_neg_integer()}]}`, where the offset is in float32s. Tensors appear in name order.
  - `Export.label_name(Scoring.label()) :: String.t()`. `{:agreement_number, :plural}` becomes `"agreement number: plural"`, and `{:word, "the"}` becomes `"word: the"`.
  - `Export.units(activations, dataset, scores, example_count) :: [map()]`. Each unit is `%{"unit", "firing_rate", "top" => [%{"label", "precision", "recall", "f1"}], "examples" => [%{"words", "position", "activation"}]}`. A unit that never fires has `"examples" => []`.
  - `mix interp.export` writes:
    - `explorer/data/model.json`: manifest plus `"vocabulary"`, `"start_token"`, `"context_length"`, `"rmsnorm_epsilon"`
    - `explorer/data/model.bin`
    - `explorer/data/sae.json`: manifest plus `"active_count"`
    - `explorer/data/sae.bin`
    - `explorer/data/neurons.json` and `explorer/data/features.json`: from `Export.units/4`, 10 examples each
    - `explorer/test/fixtures.json` (Task 2 defines its contents)

- [ ] **Step 1: Write the failing tests** in `export_test.exs`:
  - `"weights round trip through the manifest and binary"`: build `%{a: 2×3 tensor, b: {4} tensor}` and call `Export.weights/1`. The manifest names are `["a", "b"]`, the shapes are `[[2, 3], [4]]` and the offsets are `[0, 6]`. Decoding the binary with `Nx.from_binary(binary, :f32)` and slicing by offset gives back both tensors exactly.
  - `"label names read as plain words"`: `label_name({:agreement_number, :plural}) == "agreement number: plural"` and `label_name({:word, "the"}) == "word: the"`.
  - `"a unit that never fires has no examples"`: take activations from `Model.init(%Model.Config{})` on `Grammar.seed(15)` with 50 trees, and zero column 3. `Export.units/4` gives unit 3 `"firing_rate" == 0.0` and `"examples" == []`. Unit 0 has 5 examples when asked for 5, strongest first.
  - `"the whole export encodes as JSON"`: `JSON.encode!` of every map above succeeds. That catches atoms and tuples left inside.
- [ ] **Step 2: Run them, and expect a failure because `Interpretability.Export` is undefined.** Run `mise exec -- mix test test/interpretability/export_test.exs`.
- [ ] **Step 3: Implement `Export` and the mix task.** The moduledoc explains, for the speaker, why the page gets raw float32s: decimals would round, and rounding would break the parity test. Use `Activations.top_examples/4`, swapping in feature activations as `hidden` to get feature examples. Unit scores come from `Scoring.score/2`. Read `model.etf` and `sae_512.etf` only.
- [ ] **Step 4: Run the tests again, and expect them all to pass. Then run `mix interp.export`.** Expect it to print the file sizes. `model.bin` should be about 70 KB, and `sae.bin` about 530 KB.
- [ ] **Step 5: Commit** with the message "Export the model, SAE and unit data for the browser explorer".

### Task 2: JS forward pass and SAE, with the parity test

**Files:**
- Create: `interpretability/explorer/js/tensors.js`, which loads a manifest plus an `ArrayBuffer` into named `{shape, data: Float32Array}` tensors
- Create: `interpretability/explorer/js/model.js`
- Create: `interpretability/explorer/js/sae.js`
- Create: `interpretability/explorer/test/parity.test.js`
- Modify: `interpretability/lib/mix/tasks/interp.export.ex` (fixtures)

**Interfaces:**
- Consumes: the Task 1 data files.
- Produces:
  - `loadTensors(manifest, arrayBuffer) -> {name: {shape, data}}`
  - `forward(model, ids, push = null) -> {logits: Float32Array[length × 70], hidden: Float32Array[length × 128]}`. Here `ids` already starts with the start token. `push` is an optional `Float32Array(128)` added to `hidden` at every position, which Task 5 uses.
  - `encode(sae, hiddenRow, activeCount) -> Float32Array(512)`, keeping every value `>=` the `activeCount`th largest after ReLU, the same as `SAE.encode/3`.
  - `fixtures.json`: `{"sentences": [{"words", "ids", "logits", "hidden", "features"}]}`. It covers the cold open `the llama who chases the dogs`, plus the first 20 evaluation sentences, all computed by Nx. `features` lists the nonzero `[index, value]` pairs per position.

- [ ] **Step 1: Make `interp.export` write `fixtures.json`.** Then write `parity.test.js` with `node:test` and `node:assert`. It reads the data and fixture files from disk with `fs`, so it needs no server.
  - `"logits match Nx within 1e-4"`: the maximum absolute difference across all fixture sentences is at most `1e-4`.
  - `"neurons match Nx within 1e-4"`: the same check on `hidden`.
  - `"active features match Nx"`: per position, the set of active indices matches. A mismatch is allowed only when the feature's value is within `1e-4` of the cutoff. Values of features in both sets agree within `1e-4`.
  - `"keeps ties at the cutoff, like Nx"`: a hand built row whose 8th and 9th largest values are equal gives 9 active features.
  - `"a forward pass takes under 50 ms"`: average over 100 runs on a 19 word sentence.
- [ ] **Step 2: Run `node --test explorer/test/` and expect it to fail, because the modules don't exist.**
- [ ] **Step 3: Implement `tensors.js`, `model.js` and `sae.js`.** Follow `Model.run/3` exactly: embeddings plus positions, RMSNorm with epsilon `1e-6` and `gain1`, one causal attention head scaled by `sqrt(32)`, residual, RMSNorm with `gain2`, `relu(· weight1 + bias1)`, `+ hidden · weight2 + bias2`, RMSNorm with `gain3`, then `· projection`. Accumulate sums in ordinary JS numbers (float64). Each file opens with a comment, in the moduledoc style, explaining the step to a reader of the code.
- [ ] **Step 4: Run `node --test explorer/test/` and expect all 5 tests to pass.** Record the largest logit difference it prints in `docs/decisions.md`.
- [ ] **Step 5: Commit** with the message "Run the forward pass in plain JS, and test it against Nx".

### Task 3: Turning typed text into word ids

**Files:**
- Create: `interpretability/explorer/js/sentence.js`
- Create: `interpretability/explorer/test/sentence.test.js`

**Interfaces:**
- Produces: `parse(text, vocabulary, startToken, contextLength) -> {ok: true, words, ids} | {ok: false, message}`.

- [ ] **Step 1: Write the failing tests:**
  - `"The llama who chases the dogs."` gives `ok`, with words `["the", "llama", "who", "chases", "the", "dogs"]`. The ids begin with the start token's id.
  - Extra spaces and `!`, `?`, `,` are ignored.
  - `"the llama who chases the cats"` gives `ok: false`, with a message naming `cats`. Use whichever word is truly missing from the 70 word vocabulary. Check it against `model.json` in the test.
  - 20 known words give `ok: false`, with a message saying at most 19 words fit.
  - An empty string gives `ok: false` with `message: "Type a sentence, or pick one below."`
- [ ] **Step 2: Run `node --test explorer/test/sentence.test.js` and expect it to fail.**
- [ ] **Step 3: Implement `parse`.**
- [ ] **Step 4: Run it again and expect it to pass.**
- [ ] **Step 5: Commit** with the message "Parse typed sentences into the model's word ids".

### Task 4: The page

**Files:**
- Create: `interpretability/explorer/index.html`, `interpretability/explorer/style.css`, `interpretability/explorer/js/main.js`
- Create: `interpretability/explorer/README.md`, explaining how to serve it locally (`python3 -m http.server -d explorer`), how to re-export, and how to run the tests

**Interfaces:**
- Consumes: everything above. `main.js` holds DOM code only. Any logic worth testing goes into the pure modules.

What the page shows, top to bottom:
1. A sentence box plus preset buttons. The cold open is the default. Other presets: `the dogs near the llama chase the cats` (or whatever valid near-subject distractor the vocabulary allows), and one past tense sentence.
2. A row of words. The selected word is outlined. When a unit is selected, each word is shaded by that unit's activation.
3. **"Next word"** for the selected word: the top 5 words with their probabilities.
4. A toggle between **SAE features** and **neurons**, then the units active at the selected word, strongest first. Each shows its activation bar, its best label, and that label's precision and recall.
5. A panel for the selected unit: its top 3 labels with precision and recall, its firing rate, and its 10 strongest evaluation examples with the firing word bolded. A dead unit shows "never fires on the evaluation sentences".

- [ ] **Step 1: Build the page.** It loads the four data files with `fetch`, then calls `forward` once per sentence change. Clicking only redraws.
- [ ] **Step 2: Check it by hand in Chrome at 375 px and at desktop width,** using the claude-in-chrome tools. Check that:
  - the cold open's selected word `dogs` predicts a singular verb first;
  - clicking a feature shades the words;
  - a dead feature shows the message;
  - an unknown word shows the message;
  - nothing scrolls sideways;
  - tap targets are at least 40 px.

  Save a screenshot of each width to the scratchpad.
- [ ] **Step 3: Check the size budget.** Run `gzip -c` on each file in `explorer/` and sum the results. The total must be under 1.5 MB. Record it in `README.md`.
- [ ] **Step 4: Commit** with the message "Add the explorer page: words, predictions, and the units behind them".

### Task 5 (nice to have, cut first): Steering in the browser

**Files:**
- Modify: `interpretability/lib/interpretability/export.ex` and `interp.export.ex`, which also write `explorer/data/steering.json` with `{"word_length": 15.75, "directions": [{"name", "description", "vector": [128 floats]}]}` for the mean difference, neuron 74 and SAE feature 29. Also add `"margins"` fixtures from `Steering.margin/4` at pushes 0, 1 and 2 for each direction, on the cold open.
- Modify: `interpretability/explorer/js/main.js`, `index.html`, `style.css`
- Create: `interpretability/explorer/js/steering.js`, with `margin(model, ids, direction, strength)`: the best plural present verb's logit minus the best singular one's, at the last word. It uses the verb lists exported in `steering.json`.
- Modify: `interpretability/explorer/test/parity.test.js`

- [ ] **Step 1: Write the failing test** `"steering margins match Nx within 1e-3"`. At push 0, the cold open margin is −5.023, matching `docs/findings.md`.
- [ ] **Step 2: Run it, and expect it to fail.**
- [ ] **Step 3: Implement it.** Add a panel with a direction picker and a push slider from 0 to 4 (in multiples of the average neuron vector length). It shows the live margin, and the top next words at the last word.
- [ ] **Step 4: Run the test and expect it to pass. Then check by hand that neuron 74 at push 2 flips the cold open to a plural verb.**
- [ ] **Step 5: Commit** with the message "Let the explorer push along a steering direction".

### Task 6: Finish

- [ ] Run the full `mix test` and `node --test explorer/test/`, and check both pass. Run `mix format --check-formatted`.
- [ ] Add a line to `docs/decisions.md` for each ruling above. Tick the explorer baseline in the talk repo's spec checkpoints.
- [ ] Push, and open a PR stacked on #6. No CI is configured. Say so in the PR body, and list the manual checks with their screenshots.
