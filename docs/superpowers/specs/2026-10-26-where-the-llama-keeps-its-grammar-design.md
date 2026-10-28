# where the llama keeps its grammar

Design for the sequel talk to "the llama who chases the dogs", and for the
code that backs it. Working subtitle: *finding monosemantic features in a
tiny transformer.*

## Intent

The AI Guild has already seen the first talk. They watched a 15,104
parameter transformer learn subject verb agreement across a distractor
noun. This talk opens that kind of model up and asks *how* it knows.

The moment the talk is built around: zoom into the model and show a unit
that has learned one clean, nameable concept (monosemanticity), then prove
it is clean with a number, then use it to change what the model writes.

The honest version of that story, and the one this talk tells, is that a
raw neuron is usually *not* clean. Small models pack more concepts than
they have neurons (superposition), so a single neuron fires on several
unrelated things. Clean concepts appear once a sparse autoencoder (SAE)
pulls them apart. This follows Anthropic's "Toy Models of Superposition"
and "Towards Monosemanticity", which used a one layer transformer.

What makes this talk more than a paper summary is the grammar. Because we
generate the training data, we know the true hidden variables behind every
token. Most interpretability work labels features by staring at examples.
We can score them against ground truth.

## The arc

The talk is on **Tuesday, December 1, 2026**, in a 45 to 60 minute slot.
Target about 40 minutes of talking, the rest for questions.

1. **Cold open.** The familiar sentence. "You saw this model get it right.
   Where does it keep the answer?"
2. **Open a neuron.** Show an MLP neuron's top activating tokens. It fires
   on several unrelated grammar concepts. It is a mess.
3. **Superposition.** Why it is a mess, shown live on the toy model from
   "Toy Models of Superposition": change feature sparsity and watch
   features start sharing dimensions.
4. **The SAE.** Train a sparse autoencoder on the activations. One concept
   slide on what it is (a wider layer that must reconstruct the
   activations while using as few units as possible at once).
5. **Clean features, scored.** Click a feature, see every token it fires
   on, and see its precision and recall against the grammar's own labels.
   Monosemanticity becomes a measurement.
6. **Steering.** Turn up the "plural subject" feature and watch the model
   write `flee` after `the llama`. This calls back to the first talk's
   central sentence. Speaker's guess at the closer; treat it as the
   strongest candidate, not a commitment.
7. **Close.** What this does and does not tell us about real models.

An optional beat, if the measurements support it: run the same neuron
analysis on the *original* first talk model and show its neurons are
comparatively clean, because 32 words in 32 dimensions puts no pressure on
it to pack. Then show the extended model, where they are not. That turns
superposition into a before and after the audience already has context
for. Only include it if the data actually shows the contrast.

## Goals

- A model, trained on an extended grammar, whose raw MLP neurons are
  visibly polysemantic.
- An SAE whose features match grammar concepts, with precision and recall
  computed against ground truth labels.
- A steering demo that measurably changes agreement behavior.
- A browser explorer (WASM preferred, JS acceptable) where the audience
  can click features and see what they fire on, ideally on their own
  phones during the talk.
- A LiveView deck, `interpretability_talk`, built the same way as
  `tiny_llm_talk`: every number on a slide comes from the model, not a
  screenshot.
- A "hard questions" document, written during the build, so the speaker
  can answer questions from experience rather than from the paper.

## Non goals

- Mixture of experts. Considered and dropped: it makes interpretability
  harder unless the router is the subject, and here it is not.
- More layers. One layer is what the reference paper used and keeps the
  thing to explain small.
- A tokenizer. One word is one token is one integer, as before. Subword
  tokens would make features fire on fragments and harder to read.
- Real text corpora. The synthetic grammar is the point.
- Re explaining attention. The audience saw the first talk.

## Anti goals

Things that would make this a failure even if it technically works:

- The speaker cannot explain a result or answer a question about it. This
  is the speaker's stated top concern and drives the "speaker
  understanding" requirements below.
- A "monosemantic" claim the audience could poke a hole in, e.g. features
  that only look clean because of cherry picked examples.
- Any change to the first talk's code, numbers, or claims.
- The browser explorer eating the schedule. It is the first thing cut if
  SAE results take longer than expected.

## Constraints

### Repo layout

- New work lives inside the `tiny_llm` repo, layered on top. **No existing
  file under `lib/`, `test/`, `docs/`, `notebooks/` or `scripts/` changes**,
  and `tiny_llm`'s own `mix.exs` keeps zero dependencies, because the
  first talk's README and slides claim exactly that.
- Builder's call (mine, flagged for the speaker): put the new code in a
  nested Mix project at `tiny_llm/interpretability/` with its own
  `mix.exs`, its own dependencies, and a path dependency on `..` if it
  needs anything from the original. That is the only way to add Nx
  without touching the original `mix.exs`. A short pointer added to
  `tiny_llm`'s README and `CLAUDE.md` is acceptable; those are additions,
  not changes to the original claims.
- The nested project gets its own `CLAUDE.md` that states which of the
  original hard constraints are relaxed (zero deps, hand written backprop)
  and which still hold (seeded reproducibility, moduledocs as talk
  material, tests mirroring `lib/`, clarity over performance).
- The deck is a new sibling repo, `interpretability_talk`, a Phoenix
  LiveView app with a path dependency on `../tiny_llm/interpretability`,
  following `tiny_llm_talk`'s structure.

### Tools

- Elixir with Nx and Axon (EXLA backend if it builds cleanly on macOS) for
  training the model and the SAE. Autodiff is fine. Purity was the first
  talk's point, not this one's. Leaving Elixir is acceptable only where
  staying in it is not an easy win.
- Every number destined for a slide reproduces from a seed, as in the
  first talk.
- macOS laptop CPU. Training the model and the SAE should each finish in a
  few minutes at most, so they can be rerun freely while tuning.

### Extended grammar

A new grammar and vocabulary in the nested project, written in the same
style as `TinyLlm.Grammar` and `TinyLlm.Vocab` (the original modules stay
as they are). Requirements:

- It keeps the original constructions, including the relative clause with
  a distractor noun, so the first talk's probe sentence still exists.
- It adds enough independent concepts that the model must pack them.
  Candidates: tense (past vs present), animacy (rocks do not chase),
  pronouns with agreement, nesting depth, subject vs object role, a second
  clause type. Expect the vocabulary to land somewhere around 80 to 150
  words; the exact number is the builder's.
- **Every generated token carries its ground truth labels** (e.g. part of
  speech, number of the governing subject, inside relative clause or not,
  tense, syntactic role). These labels are what features get scored
  against. They are never shown to the model.

### Model

- Same family as the first talk: decoder only, one attention head is fine,
  one block, learned positional embeddings, RMSNorm, MLP.
- The residual stream is small enough relative to the number of concepts
  that raw MLP neurons come out polysemantic. Tune the grammar size and
  `d_model` (16 is a reasonable first try) until they do. "Neuron" in the
  talk means an MLP hidden unit.
- It must still get agreement across a distractor well enough that the
  cold open is honest. Report the same style of probe accuracy table as
  the first talk.

### SAE

- Trained on the MLP hidden activations, following "Towards
  Monosemanticity": one hidden layer wider than the input, ReLU, trained
  to reconstruct with a sparsity penalty. L1 versus TopK is the builder's
  call; record why.
- Report reconstruction quality, how much the model's loss suffers when
  the SAE's reconstruction replaces the real activations, dead features,
  and features that fire on nearly everything.
- Feature scoring: for every feature and every grammar label, precision
  and recall of "feature fires" against "label is true". A feature counts
  as monosemantic for a label above a threshold the builder sets and
  states (0.9 on both is a reasonable start).
- At least one experiment whose result the speaker does not know
  beforehand. The suggested one: train SAEs of two or more widths and see
  whether a feature such as "plural subject" splits into finer features.

### Steering

- Add a multiple of a feature's decoder direction to the activations
  during generation and measure the effect on agreement probes (e.g. how
  often `the llama who chases the dogs` is followed by a plural verb as
  the multiple grows). Report it as a curve, not one cherry picked
  sample.

### Browser explorer

- A static page that runs the model's forward pass client side and shows,
  for a sentence the viewer types or picks, which SAE features fire on
  each token, with each feature's grammar label and score. Steering in
  the browser is a nice to have.
- **The route is Popcorn** (Software Mansion's Elixir in the browser,
  which runs AtomVM compiled to WASM; v1.0.0 on hexdocs at the time of
  writing). The speaker asked for it, and it is the payoff for this
  audience: Elixir running the model on their phones.
  - Nx does not run on AtomVM, so the browser forward pass is plain Elixir
    over lists of floats, in the style of the first talk's
    `TinyLlm.Tensor`. That is a deliberate callback: the hand written
    math from the first talk comes back as the thing that runs in the
    browser. Reuse `TinyLlm.Tensor` directly if AtomVM's standard library
    supports what it calls; otherwise write a small copy in the nested
    project.
  - A test must check that the plain Elixir forward pass matches the Nx
    forward pass on the same weights, to a stated tolerance, so the
    browser shows the same model the slides describe.
  - Popcorn runs in a sandboxed iframe with a JS bridge; the page's
    drawing (token highlights, feature lists) can be plain JS or HTML fed
    by that bridge.
  - Known costs, from Software Mansion's own write up: a bundle over 3MB
    once the Erlang and Elixir libraries are included, a subset of the
    standard library, and "not for production" maturity. Fine for a talk
    demo; plan around them.
- Time box the Popcorn attempt to about two days. If it is not working by
  then, the fallback is plain JS reading the same exported weights. ONNX
  Runtime Web or a Rust/Zig WASM build are not worth trying in between on
  this schedule.
- Weights and SAE dictionaries are exported from the Elixir side to a
  simple format the page loads.
- The deck either embeds the explorer or links it with a QR code. My
  guess is both.

### Speaker understanding

These are requirements, not extras, because of the anti goal above.

- Build order starts with the toy superposition model, before the llama
  model, so the speaker has watched superposition happen before needing
  to explain it.
- A `docs/hard-questions.md` (in the nested project) with honest answers,
  updated as results come in. It must cover at least:
  - Why an SAE instead of PCA? (The first talk used PCA.)
  - Aren't you just finding the concepts you put into the grammar? The
    answer to defend: the grammar supplies labels to check against, but
    the model never saw them; it had to discover them.
  - What about dead features, or features that fire on everything?
  - How do we know a feature belongs to the model and is not an invention
    of the SAE?
  - Does any of this work on real models?
- Every module's `@moduledoc` explains the concept, as in the first repo,
  so the code is reading material for the speaker.

### Live versus precomputed (builder's call, flagged for the speaker)

- The llama model and its SAEs are **not** trained on stage. The deck
  loads committed checkpoints and precomputed results (feature scores,
  steering curves, SAE width comparison). This keeps EXLA out of the
  deck's Docker build and makes EXLA safe for training.
- The superposition toy model is small enough (a handful of features in
  two dimensions) to train live in the deck on `Nx.BinaryBackend`, which
  is deterministic. If it is too slow for a slider, the slider scrubs a
  precomputed sparsity sweep instead.
- "Reproduces from a seed" means: rerunning the training script with the
  same seed on the same machine gives the same reported numbers to the
  precision shown on slides. Bitwise identity across machines, which the
  first deck checks to 1.0e-9, is not required here.

### Hosting the explorer (builder's call, flagged for the speaker)

- The explorer is served as static files from the deck's existing
  deployment pattern (Fly, as `tiny_llm_talk` does), at a public URL, with
  a QR code slide. The code and weights are public, like the first repo.
- Acceptance includes a phone over mobile data loading it in under about
  ten seconds and then responding to a click within a second. Popcorn's
  bundle of a few MB fits that; a bundle that does not counts as the
  Popcorn attempt failing, and the fallback is JS.

## Decided

- Title: "where the llama keeps its grammar". Subtitle working, open to
  change.
- Audience: AI Guild, who saw the first talk.
- No MoE, no extra layers, no tokenizer.
- New code layered inside `tiny_llm` without changing existing code.
- Deck repo: `interpretability_talk`, LiveView, same pattern as before.
- Purity is optional; Elixir where it is an easy win.
- Browser explorer runs on Popcorn; JS is the fallback.
- Date: Tuesday, December 1, 2026.
- If features do not come out clean, the talk goes ahead anyway. The
  speaker's words: "the value is in the journey." Lower the threshold and
  state it plainly, framed as how far SAEs get on a model where we know
  the answer. Results never get dressed up to look cleaner than they are.
- This design document lives in `interpretability_talk`. The model code
  lives in `tiny_llm/interpretability/`.

## Left to the builder

- The exact extended grammar and vocabulary.
- `d_model`, MLP width, SAE width, sparsity method, and thresholds, all
  tuned against the goals above and recorded with the reasons.
- How to wire Popcorn into the page, within the time box.
- Deck slide structure, timed against the arc above in the same style as
  `tiny_llm/docs/talk-outline.md`.

## Open risks

- **Five weeks.** From October 28 to December 1, around a day job.
  Interpretability work has no natural end. Order of cuts if time runs
  short: the optional before and after beat, then the SAE width
  experiment, then the browser explorer (the deck alone still tells the
  story), and never the scored features or steering. Suggested
  checkpoints: toy model and extended grammar by November 6; trained model
  with polysemantic neurons and a first SAE by November 13; scoring and
  steering by November 20; explorer and deck by November 27, leaving the
  last days for rehearsal.
- **Neurons might come out clean anyway**, or the SAE might not find clean
  features. Both are tunable through grammar size and `d_model`. Per the
  decision above, an imperfect result still ships, honestly framed, but
  slides are drafted against real measurements, never ahead of them.
