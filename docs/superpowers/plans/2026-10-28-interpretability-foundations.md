# Interpretability Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tiny_llm/interpretability/`: the toy superposition model, the labeled extended grammar, an Nx llama model trained on it, a neuron analysis, and a first scored SAE. This covers the November 6 and November 13 checkpoints.

**Architecture:** A nested Mix project inside `tiny_llm`. The grammar generates parse trees. A labeler flattens each tree into tokens that carry ground truth labels. Training uses `Nx.Defn` autodiff with a Polaris Adam loop. One scoring module rates both raw neurons and SAE features against the labels with precision and recall, so the "messy neuron" and the "clean feature" are measured the same way. Mix tasks produce checkpoints and print reports, which the speaker reads before the next task begins.

**Tech Stack:** Elixir 1.18.4, OTP 28 (pinned with `mise`). Nx `~> 1.0`, Polaris `~> 0.2.0`, EXLA `~> 1.0` (only in `:dev` and `:test`).

**Spec:** `docs/superpowers/specs/2026-10-26-where-the-llama-keeps-its-grammar-design.md`, in `interpretability_talk`.

**Later plans (not this one):** steering and the SAE width experiment (November 20), the browser explorer (JS baseline, Popcorn stretch), and the deck (November 27).

## Global Constraints

- No existing file in `tiny_llm` changes, apart from a short pointer added to `README.md` and `CLAUDE.md`. `tiny_llm/mix.exs` keeps zero dependencies.
- `d_model` is 32 and the MLP width is 128 (4 × `d_model`), the first talk's values. Neither is tuned.
- One block, one attention head, learned positional embeddings, pre-norm RMSNorm, ReLU MLP, untied unembedding. Parameter names match `TinyLlm.Transformer`: `embeddings`, `positions`, `gain1`, `query_weight`, `key_weight`, `value_weight`, `output_weight`, `gain2`, `weight1`, `bias1`, `weight2`, `bias2`, `gain3`, `projection`.
- One word is one token. There is no tokenizer.
- Every number destined for a slide reproduces from a seed. On `Nx.BinaryBackend` that means bitwise. On EXLA it means the same machine and the precision shown on slides.
- Training the model and each SAE finishes in a couple of minutes at most on the M1 Max CPU. EXLA runs on the CPU. EMLX is out of scope.
- Every module's `@moduledoc` explains the concept, because the code is reading material for the speaker.
- Tests mirror `lib/`. Public and private functions are separated by `## PRIVATE FUNCTIONS`. No single letter or abbreviated variable names, even in math (`weights`, not `w`). No em dashes or dashes in prose.
- Library code never names `EXLA` at compile time. `Interpretability.Compiler.options/0` picks EXLA when it is loaded, so the deck can depend on this project without EXLA.
- Every builder decision goes in `docs/decisions.md` with its reason, in the task where it is made.
- **Half teaching, half building.** The speaker is learning these concepts during the build and the audience is software engineers without a statistics background. Before each task that introduces a concept (superposition, labels as ground truth, MLP neurons, precision and recall, SAEs), explain it in plain terms with a concrete example from this grammar and check understanding before writing code. Prefer intuition over statistics.
- **Speaker checkpoints:** a step marked **SPEAKER** stops the run. Ask the speaker for a prediction, record it in `docs/findings.md`, and only then run the experiment. Afterwards write the result next to the prediction, with a two or three sentence explanation the speaker could say out loud.

## Review Focus

1. **Labels are off by one from activations.** The activation at input position t must be paired with the labels of input token t, where position 0 is `<start>`. Pinned by `Dataset` test "labels align with input tokens".
2. **Sentences longer than the context.** Every generated sentence must fit in 20 tokens, including `.`. Pinned by `Grammar` test "every sentence fits the context".
3. **Dead units and empty labels in scoring.** A unit that never fires, or a label value with no positives, must score 0.0 and never NaN. Pinned by `Scoring` tests.
4. **The cold open sentence.** `the llama who chases the dogs` must be producible by the grammar, and the trained model must rank `flees` above `flee` after it. Pinned by `Grammar` test "can produce the probe sentence" and by the `interp.train` gate.
5. **Rerunning the same seed.** It must give the same corpus, and on BinaryBackend the same trained toy weights. Pinned by `Grammar` and `Toy` determinism tests.

---

### Task 1: Scaffold the nested project and check the toolchain

**Files:**
- Create: `tiny_llm/interpretability/{mix.exs,mise.toml,.formatter.exs,.gitignore,CLAUDE.md,README.md}`
- Create: `tiny_llm/interpretability/lib/interpretability/compiler.ex`
- Create: `tiny_llm/interpretability/docs/{decisions.md,findings.md,hard-questions.md}`
- Modify: `tiny_llm/README.md` and `tiny_llm/CLAUDE.md` (append a short "Sequel: interpretability/" pointer only)
- Test: `tiny_llm/interpretability/test/interpretability/compiler_test.exs`

**Interfaces:**
- Produces: `Interpretability.Compiler.options() :: keyword()`, which returns `[compiler: EXLA]` when `Code.ensure_loaded?(EXLA)` is true and `[]` otherwise. Every `Nx.Defn.jit` call in later tasks passes these options.

- [ ] **Step 1:** Create the worktree with superpowers:using-git-worktrees in `~/src/talks/tiny_llm`, off `main`. Run `mix new interpretability --app interpretability --module Interpretability` inside it.
- [ ] **Step 2:** In `mix.exs`, add deps `{:nx, "~> 1.0"}`, `{:polaris, "~> 0.2.0"}`, `{:exla, "~> 1.0", only: [:dev, :test]}`. Add `mise.toml` pinning `erlang = "28"` and `elixir = "1.18.4-otp-28"`, and `priv/results/` to `.gitignore`. Checkpoints stay committed.
- [ ] **Step 3:** Write a test that jits `fn tensor -> Nx.multiply(tensor, 2) end` with `Compiler.options()` and asserts that the result for `Nx.tensor([1.0, 2.0])` is `[2.0, 4.0]`.
- [ ] **Step 4:** Run `mix deps.get && mix test`. Expected: PASS, with EXLA compiling. **If EXLA fails to build, do not debug it.** Remove it, rerun on BinaryBackend, and record the failure in `decisions.md`.
- [ ] **Step 5:** Write `CLAUDE.md`. It relaxes zero deps and hand written backprop. It keeps seeded reproducibility, moduledocs as talk material, tests mirroring `lib/`, and clarity over performance. It points to the spec and this plan.
- [ ] **Step 6:** Seed `hard-questions.md` with the spec's questions. There is no PCA question, because the audience has no statistics background. Answers start as "pending a result". Seed `decisions.md` with "Nx.Defn and Polaris instead of Axon: steering and SAE splicing need to replace the MLP hidden activations mid forward pass, which is one function argument in defn and a hook in Axon."
- [ ] **Step 7:** Commit with the message "Scaffold the interpretability project so sequel code stays out of the zero dependency original".

### Task 2: Toy model of superposition

**Files:**
- Create: `lib/interpretability/toy.ex`, `lib/mix/tasks/interp.toy.ex`
- Test: `test/interpretability/toy_test.exs`

**Interfaces:**
- Produces: `%Interpretability.Toy.Config{feature_count: 5, dimension_count: 2, importance_decay: 0.8, sparsity: 0.0, batch_size: 1024, steps: 3_000, learning_rate: 1.0e-2, seed: 1234}`
- Produces: `Toy.train(Config.t()) :: %{weights: Nx.Tensor.t() (2×5), bias: Nx.Tensor.t() (5), loss: float()}`
- Produces: `Toy.represented(result, threshold \\ 0.5) :: non_neg_integer()`, the number of feature columns whose norm exceeds the threshold.
- Produces: `Toy.sweep([float()], Config.t()) :: [%{sparsity: float(), weights: Nx.Tensor.t(), bias: Nx.Tensor.t(), loss: float(), represented: non_neg_integer()}]`

The model follows "Toy Models of Superposition". Each feature is zero with probability `sparsity` and uniform on [0, 1] otherwise. The reconstruction is `relu(transpose(weights) · weights · features + bias)`. The loss is `sum over features of importance_i × (feature_i − reconstruction_i)²`, averaged over the batch, with `importance_i = importance_decay^i`. It runs on BinaryBackend, so the deck can train it live.

- [ ] **Step 1:** Write the tests:
  - `represented(train(%Config{sparsity: 0.0})) == 2`. Dense features: only the two most important ones fit.
  - `represented(train(%Config{sparsity: 0.95})) >= 4`. Sparse features: they share the two dimensions.
  - Training twice with the same seed gives `Nx.to_list` equal weights.
- [ ] **Step 2:** Run `mix test test/interpretability/toy_test.exs`. Expected: FAIL, because `Toy` is undefined.
- [ ] **Step 3:** Implement `Toy`, using `Nx.Random.key(seed)` for data and initialization and Polaris `adam` for training. The moduledoc explains superposition in plain words, as material for the speaker.
- [ ] **Step 4:** Run the tests. Expected: PASS. If the 0.95 case gives 3, report it rather than tuning silently. That is a finding.
- [ ] **Step 5:** Implement `mix interp.toy`. It sweeps `[0.0, 0.5, 0.8, 0.9, 0.95, 0.99]`, prints a table of sparsity, represented count, loss, and wall time, and writes `priv/results/toy_sweep.etf`.
- [ ] **Step 6: SPEAKER.** Ask: "With 5 features and 2 dimensions, how many features does the model keep at 0% sparsity, and how many at 95%? What shape do they make?" Record the answer. Run the task. Show the weight columns as 2D points and explain the pentagon, or whatever shape actually appears.
- [ ] **Step 7:** Commit with the message "Add the toy superposition model so the speaker has watched features share dimensions before explaining it".

### Task 3: Extended vocabulary

**Files:**
- Create: `lib/interpretability/vocab.ex`
- Test: `test/interpretability/vocab_test.exs`

**Interfaces:**
- Produces: `Vocab.words/0`, `size/0`, `word_to_id/1`, `id_to_word/1`, `encode/1`, `decode/1`, all with the same contracts as `TinyLlm.Vocab`. Plus category accessors: `determiners/0`, `nouns(animacy, number)`, `pronouns/0`, `transitive_verbs(tense, number)`, `intransitive_verbs(animacy, tense, number)`, `copula(tense, number)`, `adjectives/0`, `relative_pronouns/0`, `prepositions/0`, `start_token/0`, `end_token/0`. Past tense forms, except the copula, ignore `number` and return the same word for both.

The vocabulary is pinned below and recorded in `decisions.md`. It has 70 words, in this id order:

| Category | Words |
| --- | --- |
| determiners | the a |
| animate nouns (singular, plural) | llama llamas, dog dogs, goose geese, fox foxes, mouse mice, cat cats, owl owls, wolf wolves |
| inanimate nouns | rock rocks, tree trees, river rivers, hill hills |
| pronouns | he she it they |
| transitive verbs, present (singular, plural) | sees see, chases chase, ignores ignore, follows follow |
| transitive verbs, past | saw chased ignored followed |
| animate intransitive verbs | flees flee, sleeps sleep, past: fled slept |
| inanimate intransitive verbs | falls fall, past: fell |
| copula | is are, past: was were |
| adjectives | big small hungry grumpy fast sleepy old wet |
| relative pronouns, conjunction, prepositions | who that, and, near behind |
| boundaries | `<start>` `.` |

- [ ] **Step 1:** Write the tests: `size() == 70`, ids round trip for every word, words are unique, `nouns(:animate, :plural)` contains `"llamas"`, `copula(:past, :plural) == ["were"]`, and `word_to_id("zebra")` raises.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the module in `TinyLlm.Vocab`'s style. The moduledoc explains why the vocabulary grew: more concepts, not more words for their own sake.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Add the extended vocabulary, sized by the concepts the grammar needs".

### Task 4: Grammar that generates parse trees

**Files:**
- Create: `lib/interpretability/grammar.ex`
- Test: `test/interpretability/grammar_test.exs`

**Interfaces:**
- Produces these tree shapes, which are plain maps:
  - Noun phrase:
    - `%{type: :simple, determiner, adjectives: [word], noun, number, animacy}`
    - `%{type: :compound, left: simple, right: simple}` (always plural and animate)
    - `%{type: :relative, head: simple, relative_pronoun, verb_phrase}`
    - `%{type: :prepositional, head: simple, preposition, object: simple}`
    - `%{type: :pronoun, pronoun, number, animacy}`
  - Verb phrase:
    - `%{type: :intransitive, verb, tense, number}`
    - `%{type: :transitive, verb, tense, number, object: noun_phrase}`
    - `%{type: :copula, verb, tense, number, adjective}`
  - Sentence: `%{subject: noun_phrase, verb_phrase: verb_phrase}`
- Produces: `Grammar.seed(integer)`, `Grammar.tree() :: sentence_tree`, `Grammar.trees(count)`, `Grammar.words(sentence_tree) :: [word]` (ends in `"."`), `Grammar.max_tokens() == 20`

Probabilities are pinned here and recorded in `decisions.md`:
- **Subject noun phrase:** simple 50, compound 12, relative 18, prepositional 12, pronoun 8.
- **Object noun phrase:** simple 80, relative 20. Only relative while depth < 2; otherwise always simple.
- **Animacy:** subjects are 80% animate; objects and prepositional objects are 70% animate.
- **Pronouns:** `he` and `she` are singular and animate, `it` is singular and inanimate, and `they` is plural with animacy drawn 80/20.
- **Verb phrase for an animate subject:** intransitive 30, transitive 40, copula 30. For an inanimate subject: intransitive (`falls`) 40, copula 60.
- **Tense:** drawn per verb phrase, present 70 and past 30.
- **Adjective phrase:** none 55, one 30, two 15.
- **Relative pronoun:** `who` for an animate head, `that` for an inanimate one.
- **Determiner:** `a` only with a singular noun.
- **Length:** resample any sentence longer than 20 tokens.

- [ ] **Step 1:** Write the tests:
  - **Determinism:** after `seed(7)`, `trees(200)` is identical on two runs.
  - **Every sentence fits the context:** after `seed(1)`, every one of 5,000 sentences has `length(words) <= 20` and ends in `"."`.
  - **Agreement holds by construction:** every present tense verb's form matches its verb phrase's `number`.
  - **Can produce the probe sentence:** `words` of the hand built tree for `the llama who chases the dogs flees .` returns exactly those words, and the generator emits at least one `:relative` subject whose head and object numbers differ within 5,000 trees.
  - **Constructions appear at their rates:** each of the five subject types appears within ±3 percentage points of its target over 5,000 trees.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the module. The moduledoc gives the rules as a BNF block, as `TinyLlm.Grammar` does, and explains which number each distractor discards.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Generate parse trees, not word lists, so every token's ground truth can be read off the tree".

### Task 5: Labeler and agreement probes

**Files:**
- Create: `lib/interpretability/labeler.ex`, `lib/interpretability/probes.ex`
- Test: `test/interpretability/labeler_test.exs`, `test/interpretability/probes_test.exs`

**Interfaces:**
- Produces: `Labeler.label(sentence_tree) :: [%{word: word, labels: labels}]`, where `labels` is a map with exactly these keys:
  - `:word`: the word itself.
  - `:part_of_speech`: one of `:determiner, :noun, :pronoun, :transitive_verb, :intransitive_verb, :copula, :adjective, :relative_pronoun, :conjunction, :preposition, :start, :end`.
  - `:word_number`: `:singular | :plural | nil`. Nouns, pronouns, and present tense verbs carry it, as do `was`, `were`, and `a`.
  - `:agreement_number`: the number the next finite verb must take, **from left context only**. It is the number of the innermost clause whose verb has not appeared yet. It becomes known at that clause's head noun, at its pronoun, or at the `and` of a compound. It is `nil` before then, and after the main verb when no clause is open.
  - `:role`: `:subject | :object | :prepositional_object | nil`, on every token of a noun phrase. For a relative clause's head, it is the head's role in the outer clause.
  - `:depth`: 0, 1, or 2, the number of enclosing relative clauses. The relative pronoun is at the new depth.
  - `:tense`: `:present | :past | nil`, on finite verbs only.
  - `:animacy`: `:animate | :inanimate | nil`, on nouns and pronouns only.
  - `:after_main_verb`: boolean, true from the main verb onward.
- Produces: `Labeler.start_labels() :: labels`, for `<start>`: `:part_of_speech` is `:start`, `:depth` is 0, `:after_main_verb` is false, `:word` is `"<start>"`, and every other key is nil.
- Produces: `Probes.from_tree(sentence_tree) :: [%{prefix: [word], number: grammatical_number, category: atom, mismatch: boolean}]`. There is one probe per present tense finite verb, with the prefix being the words before it. Categories are `:simple_subject, :compound_subject, :pronoun_subject, :relative_distractor, :prepositional_distractor, :inside_relative, :object_relative`. `mismatch` is true when the nearest preceding noun's number differs from the expected number.
- Produces: `Probes.accuracy(predictor, probes) :: %{{category, mismatch} => float}`. A `predictor` is `([word] -> logits list)`. A probe is correct when the highest ranked present tense verb or copula has the right number, as in `TinyLlm.Eval.agreement/2`.

- [ ] **Step 1:** Write labeler tests on the hand built tree for `the llama who chases the dogs flees .`:
  - `:agreement_number == [nil, :singular, :singular, :singular, :singular, :singular, nil, nil]`
  - `:depth == [0, 0, 1, 1, 1, 1, 0, 0]`
  - `:role == [:subject, :subject, nil, nil, :object, :object, nil, nil]`
  - `:after_main_verb == [false, false, false, false, false, false, true, true]`

  Then on `the fox sees the dogs who flee .`:
  - `:agreement_number == [nil, :singular, nil, nil, nil, :plural, nil, nil]`
  - `:role` of `dogs` is `:object`
  - `:depth` of `flee` is 1

  Then the property test: for 2,000 seeded trees, every token's labels map has exactly the nine keys.
- [ ] **Step 2:** Write probe tests:
  - `from_tree` on the first sentence returns `[%{prefix: ~w(the llama who), number: :singular, category: :inside_relative, mismatch: false}, %{prefix: ~w(the llama who chases the dogs), number: :singular, category: :relative_distractor, mismatch: true}]`.
  - A past tense main verb yields no probe.
  - A stub predictor that always prefers singular scores 1.0 on every singular probe.
- [ ] **Step 3:** Run them. Expected: FAIL.
- [ ] **Step 4:** Implement both modules. Record the left context definition of `:agreement_number` in `decisions.md`, and why: a label that knows the future would let features score well for information the model cannot have.
- [ ] **Step 5:** Run them. Expected: PASS.
- [ ] **Step 6:** Commit with the message "Label every token from the parse tree so features can be scored against ground truth". **This is the November 6 checkpoint.** Show the speaker 10 labeled sentences.

### Task 6: Dataset

**Files:**
- Create: `lib/interpretability/dataset.ex`
- Test: `test/interpretability/dataset_test.exs`

**Interfaces:**
- Consumes: `Grammar.words/1`, `Labeler.label/1`, `Labeler.start_labels/0`, `Vocab.encode/1`
- Produces: `Dataset.build([sentence_tree], context_length \\ 20) :: %{tokens: Nx.Tensor.t() (n×20 s64), targets: Nx.Tensor.t() (n×20 s64), mask: Nx.Tensor.t() (n×20 f32), labels: [[labels]], trees: [sentence_tree]}`. The input is `<start>` followed by the sentence without its final `.`. The targets are the sentence. Padding uses the `.` id with a mask of 0. `labels` holds one list per sentence, covering only the unpadded input positions.
- Produces: `Dataset.batches(dataset, batch_size, seed) :: Enumerable.t()`, an infinite stream of `%{tokens, targets, mask}` that reshuffles with `:rand` (`:exsss`) every epoch.

- [ ] **Step 1:** Write the tests:
  - **Labels align with input tokens:** for every sentence, `length(labels) == sum(mask row)`. The labels at position 0 are `start_labels()`, and the `:word` label at position t equals `Vocab.id_to_word(tokens[t])` for every unmasked t.
  - Targets are the inputs shifted left by one.
  - The same seed gives the same first 3 batches.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the module.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Build padded tensors whose labels stay aligned with input positions".

### Task 7: Trainer and checkpoints

**Files:**
- Create: `lib/interpretability/trainer.ex`, `lib/interpretability/checkpoint.ex`
- Test: `test/interpretability/trainer_test.exs`, `test/interpretability/checkpoint_test.exs`

**Interfaces:**
- Produces: `Trainer.run(params :: map(), batches :: Enumerable.t(), loss_function :: (map(), map() -> Nx.Tensor.t()), options) :: {map(), [%{step: integer, loss: float}]}`. The options are `steps`, `learning_rate`, `log_every`, and `after_step: (map() -> map())`, which defaults to identity. It uses Polaris `adam`, and jits the step with `Compiler.options()`.
- Produces: `Checkpoint.save(term, path)` and `Checkpoint.load(path)`. Save moves every tensor to `Nx.BinaryBackend`, then calls `:erlang.term_to_binary`.
- Toy (Task 2) may move to `Trainer.run` here if that is simpler.

- [ ] **Step 1:** Write the tests: fitting `y = 3x + 1` from 256 points gets loss below 1.0e-3 within 500 steps. An `after_step` that zeroes a parameter is applied. A checkpoint round trip of `%{weights: tensor, config: %{seed: 1}}` returns equal values.
- [ ] **Step 2:** Run them, implement, and run them again. Expected: FAIL, then PASS.
- [ ] **Step 3:** Commit with the message "Share one Adam training loop between the model and the SAE".

### Task 8: The llama model

**Files:**
- Create: `lib/interpretability/model.ex`, `lib/mix/tasks/interp.train.ex`
- Test: `test/interpretability/model_test.exs`

**Interfaces:**
- Produces: `%Model.Config{vocabulary_size: Vocab.size(), d_model: 32, hidden_width: 128, context_length: 20, seed: 1234, learning_rate: 3.0e-3, batch_size: 128, steps: 4_000, training_corpus_size: 20_000, evaluation_corpus_size: 2_000}`
- Produces: `Model.init(Config.t()) :: params`, using the parameter names in Global Constraints.
- Produces: `Model.forward(params, tokens, options \\ []) :: logits (b×c×v)`. `options[:edit_hidden]` is a defn compatible function from the MLP hidden tensor (b×c×128) to a tensor of the same shape, applied after the ReLU. It defaults to identity. SAE splicing (Task 11) and steering (next plan) rely on it.
- Produces: `Model.hidden(params, tokens) :: Nx.Tensor.t() (b×c×128)`, the post ReLU MLP activations.
- Produces: `Model.loss(params, batch, options \\ []) :: scalar`, the masked mean cross entropy, which passes `options` through to `forward`.
- Produces: `Model.predictor(params) :: ([word] -> [float])`, for `Probes.accuracy/2`.

- [ ] **Step 1:** Write the tests:
  - **Shapes:** logits are {2, 20, 70} and hidden is {2, 20, 128}.
  - **Causality:** changing the token at position 5 leaves the logits at positions 0 to 4 unchanged, to within 1.0e-6.
  - `edit_hidden: &Function.identity/1` equals the default.
  - `edit_hidden` that zeroes the hidden tensor changes the logits.
  - The initial loss is within 0.5 of `ln(70)`.
  - 200 steps on a 500 sentence corpus lowers the loss by at least 30%.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the forward pass from `TinyLlm.Transformer`'s and `TinyLlm.Block`'s moduledocs, in Nx: embed plus positions, `rmsnorm` with `gain1`, causal single head attention, residual, `rmsnorm` with `gain2`, the MLP with `edit_hidden` applied to the hidden tensor, residual, `rmsnorm` with `gain3`, then the projection.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Implement `mix interp.train`. It seeds the grammar, builds the training and evaluation datasets, trains, and saves `priv/checkpoints/model.etf` as `%{config, params, history}`. It prints the evaluation loss, the probe accuracy table by `{category, mismatch}`, the wall time, and the rank of `flees` against `flee` after `the llama who chases the dogs`.
- [ ] **Step 6:** **Gate:** relative clause distractor accuracy with `mismatch: true` must be at least 0.95, and `flees` must outrank `flee`. If the gate fails, raise `steps` first, then `training_corpus_size`, and record what changed. Do not touch `d_model`.
- [ ] **Step 7: SPEAKER.** Show the probe table. Ask which category they expect to be the weakest, and why, before revealing it.
- [ ] **Step 8:** Commit the code and the checkpoint with the message "Train the llama model on the extended grammar so there is a model to open up".

### Task 9: Scoring

**Files:**
- Create: `lib/interpretability/scoring.ex`
- Test: `test/interpretability/scoring_test.exs`

**Interfaces:**
- Produces: `Scoring.indicators([labels]) :: %{{key, value} => Nx.Tensor.t() (N, u8)}`, built from every non-nil label value. A `{key, value}` pair with zero positives is left out.
- Produces: `Scoring.score(activations :: Nx.Tensor.t() (N×U), [labels], options \\ [threshold_fraction: 0.2]) :: [%{unit: integer, firing_rate: float, best: %{label: {key, value}, precision: float, recall: float, f1: float}}]`. A unit fires on a token when its activation is greater than `threshold_fraction` × that unit's maximum over the dataset. `best` is the label with the highest F1.
- Produces: `Scoring.monosemantic?(score, threshold \\ 0.9) :: boolean`, true when both precision and recall are at least the threshold.

- [ ] **Step 1:** Write the tests:
  - A unit equal to a label's indicator scores precision 1.0 and recall 1.0 for that label.
  - A unit that fires on the union of two equally sized labels scores precision 0.5 for either.
  - An all zero unit has firing rate 0.0, precision 0.0, and recall 0.0, with no NaN.
  - A label value with no positives does not appear among the indicators.
- [ ] **Step 2:** Run them, implement, and run them again. Expected: FAIL, then PASS.
- [ ] **Step 3:** Record in `decisions.md` that the same relative firing threshold (0.2 of the unit's maximum) is applied to neurons and to SAE features. The reason is that a raw ReLU neuron "fires" at any positive value, which would rig the comparison against it.
- [ ] **Step 4:** Commit with the message "Score units against ground truth labels with precision and recall".

### Task 10: Neuron analysis

**Files:**
- Create: `lib/interpretability/activations.ex`, `lib/mix/tasks/interp.neurons.ex`
- Test: `test/interpretability/activations_test.exs`

**Interfaces:**
- Consumes: `Model.hidden/2`, `Dataset`, `Scoring.score/3`
- Produces: `Activations.collect(params, dataset) :: %{hidden: Nx.Tensor.t() (N×128), labels: [labels], contexts: [{sentence_index, position}]}`. It covers unpadded positions only, in dataset order.
- Produces: `Activations.top_examples(activations, dataset, unit, count) :: [%{activation: float, words: [word], position: integer}]`

- [ ] **Step 1:** Write the tests. On a tiny model, `N` equals the total of the mask. Row `i` of `hidden` equals `Model.hidden` at `contexts[i]`. `top_examples` is sorted in descending order.
- [ ] **Step 2:** Run them, implement, and run them again. Expected: FAIL, then PASS.
- [ ] **Step 3: SPEAKER.** Ask: "Of 128 neurons, how many will be monosemantic at 0.9 precision and recall?" Record the answer.
- [ ] **Step 4:** Implement and run `mix interp.neurons`. It loads the model, collects activations on the evaluation set, and scores them. It prints the monosemantic count at 0.9, a histogram of the best F1, and the 3 neurons with the lowest best F1, with their top 10 examples and their top 3 labels. It writes `priv/results/neurons.etf`.
- [ ] **Step 5:** Explain the result to the speaker. If the neurons are mostly clean, that is the finding the spec plans for. Write it up in `findings.md`, and do not tune anything to make the neurons messy.
- [ ] **Step 6:** Commit with the message "Measure how clean the raw neurons are, using the same yardstick the SAE will face".

### Task 11: Sparse autoencoder

**Files:**
- Create: `lib/interpretability/sae.ex`, `lib/mix/tasks/interp.sae.ex`
- Test: `test/interpretability/sae_test.exs`

**Interfaces:**
- Consumes: `Trainer.run/4` with `after_step`, `Model.loss/3` with `edit_hidden`, `Activations.collect/2`, `Scoring.score/3`
- Produces: `%SAE.Config{input_width: 128, width: 512, active_count: 8, learning_rate: 1.0e-3, batch_size: 1024, steps: 5_000, seed: 1234}`
- Produces: `SAE.init(Config.t(), activations :: Nx.Tensor.t()) :: %{encoder, encoder_bias, decoder, decoder_bias}`. The decoder rows have unit norm, and `decoder_bias` starts at the activation mean.
- Produces: `SAE.encode(sae, input, active_count) :: Nx.Tensor.t()`. It computes `relu(encoder · (input − decoder_bias) + encoder_bias)`, then keeps the top `active_count` values per row with `Nx.top_k` and zeroes the rest.
- Produces: `SAE.decode(sae, features) :: Nx.Tensor.t()`, which computes `features · decoder + decoder_bias`.
- Produces: `SAE.loss(sae, input, active_count) :: scalar`, the reconstruction MSE divided by the input variance.
- Produces: `SAE.normalize_decoder(sae) :: sae`, used as `after_step`.
- Produces: `SAE.report(sae, model_params, dataset, activations) :: %{variance_explained, loss_original, loss_spliced, loss_zero_ablated, loss_recovered, dead_count, dense_count}`. Here `loss_recovered = (zero_ablated − spliced) / (zero_ablated − original)`. Dead means the feature never fires on the evaluation set, and dense means its firing rate is above 0.5.

- [ ] **Step 1:** Write the tests:
  - At most `active_count` features are nonzero per row.
  - Decoder rows have norm 1 after `normalize_decoder`, to within 1.0e-5.
  - **Recovery on synthetic data:** generate 32 dimensional inputs, each a sum of 3 of 16 random unit directions. Train with width 64 and active count 3. `variance_explained` must exceed 0.9.
  - In `report`, an SAE whose `decode(encode(x))` is the identity on the data gives `loss_recovered` of about 1.0.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the module. Record **TopK over L1** in `decisions.md`: the sparsity is set directly as "at most 8 features on at once", which is one sentence on a slide. There is no penalty coefficient to tune, and no shrinkage. The cost is a departure from "Towards Monosemanticity", which used L1, and `hard-questions.md` should say so. The moduledoc explains the SAE as the spec's concept slide does.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5: SPEAKER.** Ask: "The neurons scored N monosemantic. How many of 512 SAE features will, and how many will be dead?" Record the answer.
- [ ] **Step 6:** Implement and run `mix interp.sae`. It trains on the training set activations, then reports and scores on the evaluation set. It saves `priv/checkpoints/sae_512.etf` with `%{config, sae, report, scores}`. It prints the report, the monosemantic count against the neuron count, and the 10 best features with their labels and precision and recall.
- [ ] **Step 7:** Fill in the `hard-questions.md` answers that this result now supports (dead and dense features, and "is the feature the SAE's invention?"), citing the numbers. Update `findings.md`.
- [ ] **Step 8:** Commit with the message "Train and score the first SAE: the November 13 checkpoint".

### Task 12: Close out

- [ ] **Step 1:** Run `mix format --check-formatted && mix compile --warnings-as-errors && mix test` in `interpretability/`, and `mix test` in `tiny_llm/` to confirm the original still passes untouched.
- [ ] **Step 2:** Run `git diff main --stat -- lib test docs notebooks scripts mix.exs` from the `tiny_llm` root. Expected: empty.
- [ ] **Step 3:** Push, open a PR against `tiny_llm`'s main, and watch CI.
