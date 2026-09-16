# Phase 5 — Property Search Ranking

Status: design approved in chat 2026-09-16; awaiting written-spec review
Date: 2026-09-16
Part of: Dubai Real Estate ML Platform (sub-project 5 of 10)
Builds on: `docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md`,
`docs/superpowers/specs/2026-09-15-phase3-price-model-design.md`

## Goal

Search the Phase 4 listings corpus from a free-text query such as
"2BR in Dubai Marina under 1.5M" and return listings in a useful order. The
system works in stages:

1. A rule parser turns the query into structured slots.
2. Two retrieval channels (semantic and full-text) fetch about 200 candidates.
3. Phase 4 duplicate clusters are collapsed.
4. A learned gradient-boosted ranker orders what is left.

There are no real users or click logs. Queries and relevance grades are
therefore generated and graded by rule, and that is stated wherever results
appear. The deliverable is the retrieval-and-ranking pipeline and an honest
evaluation of it. It makes no claim about real user satisfaction.

## Decisions (user, 2026-09-16)

- **Relevance labels: rule-graded synthetic queries.** Queries are generated
  from real listing attributes and graded 0–3 by constraint match. Simulated
  clicks and LLM-judged relevance were considered and declined.
- **Query understanding: rule parser plus embeddings.** A deterministic slot
  parser handles the structured parts. It reuses Phase 2's `dld.area_aliases`.
  The rest of the query goes to MiniLM semantic search. A learned slot tagger
  and LLM parsing were declined.
- **Phase 4 integration: collapse duplicates and demote flagged listings.**
  Search shows one listing per detected duplicate cluster. The predicted
  Phase 4 fraud flags are ranker features.
- **Approach A: two-stage retrieval with a GBDT ranker.**
  - Champion: XGBoost `rank:ndcg` on the GPU.
  - Challenger: LightGBM `lambdarank`.
  - Both are compared against three non-learned baselines.
  - A cross-encoder challenger was declined.
- **Hosting (affects later phases):** local Docker Compose plus a Cloudflare
  Tunnel replaces Cloud Run. Phase 5 has no hosting work.

## Source data

These inputs are read-only in this phase:
- `listings.listings`: 20,000 synthetic listings over real DLD sales. Fields:
  area, building, project, property type and sub-type, bedrooms, size, asking
  price and posting date.
  - Amenities appear only in `description`, drawn from the fixed list
    `listings.text.AMENITIES`.
- `listings.listing_embeddings.text_embedding`: MiniLM `vector(384)` with an
  HNSW cosine index. Queries are embedded with the same model. Listings are
  not re-embedded.
- The latest `listings.detect_runs` row, and for that run:
  `listings.duplicate_pairs` where `decision`, and `listings.fraud_flags`.
- `dld.areas` and `dld.area_aliases`, from Phase 2.
- `models:/zestimator-price@champion`, from Phase 3. It is used only for the
  value features.
- The ground-truth columns `fraud_label` and `dup_group_id` are used **only**
  by grading and evaluation, never by features, retrieval or the engine.

## Architecture

A new top-level package, `search/`. It has one module per job: pure functions
first, database and model code at the edges.

| Module | Job |
|---|---|
| `config.py` | `SearchConfig` (all constants below), `FEATURES` tuple, slot names |
| `queries.py` | Generate synthetic queries with their true slots and a template id |
| `grade.py` | Grade a candidate 0–3 against the true slots |
| `parse.py` | `parse(text, lexicon) -> ParsedQuery` (pure) |
| `lexicon.py` | Load the area, building and project lexicon from Postgres |
| `retrieve.py` | Semantic and full-text channels, RRF fusion, duplicate collapse |
| `features.py` | Candidate features from the parsed query and the listing |
| `train.py` | Fit the XGBoost and LightGBM rankers with Optuna; register the winner |
| `evaluate.py` | Metrics, baselines, bootstrap CIs, MLflow logging |
| `engine.py` | `search(conn, text, k=10) -> SearchResult` for Phase 6 |
| `__main__.py` | CLI `python -m search queries\|train\|evaluate\|query` |
| `sql/schema.sql` | `search` schema and the full-text column and index |

`search/` imports from `listings/` (the embedder and the price-predictor
loader) and from `ingestion/` (`DbSettings`). Nothing in `listings/` imports
`search/`.

### New dependencies

None. Every library needed is already in the project: xgboost, lightgbm,
optuna, sentence-transformers, polars, psycopg, mlflow and matplotlib.

## Queries (queries.py, deterministic under `--seed`)

- **Count:** `n_queries = 6000` by default.
- **Seeding:** each query is seeded from a real listing sampled uniformly. Its
  slots are drawn from that listing, so every fully specified query has at
  least one grade-3 answer.
- **Slots:** each query fills a random subset of:
  - area: the official name, or an alias from `dld.area_aliases` with
    probability 0.4;
  - building or project name (rare, p=0.1);
  - bedrooms;
  - property type;
  - budget: max only, or a min–max range around the seed price with random
    headroom of 0–30%;
  - minimum size;
  - 0–2 amenities taken from the seed listing's description.
- **Query kinds**, which are stored:
  - `specified`: 3 or more slots, 70% of queries;
  - `vague`: 1–2 slots plus free words such as "family", "investment" or
    "quiet", 20%;
  - `no_match`: slots combined so that no listing satisfies them all, 10%.
    These are checked against the corpus at generation time.
- **Templates:** about 30 phrasing templates, each with an integer
  `template_id`, render the slots. Examples: "2BR in {area} under {budget}",
  "looking for a {beds} bedroom {type} around {area}, max {budget}", "{type}
  {area} {amenity}". Numbers are rendered in several forms: "1.5M",
  "1,500,000", "AED 1.5 million", "900k". Bedrooms appear as "2BR", "2 bed",
  "two bedroom" or "studio".
- **Output:** `search.queries`, with `query_id`, `text`, `template_id`, `kind`,
  `split`, `true_slots jsonb` and `seed_listing_id`.
- **Splits are assigned by `template_id`:**
  - about 60% of templates go to `train`, 20% to `tune` and 20% to `report`;
  - the assignment is deterministic under the seed;
  - every split contains every query kind;
  - no template appears in two splits;
  - no exact query text appears in two splits.

## Grading (grade.py)

`grade(true_slots, listing) -> int in 0..3`, a pure function:
- **3:** every stated slot holds, and the price lies within `[min, max]` where
  given.
- **2:** exactly one near miss, with every other slot holding. A near miss is
  one of:
  - bedrooms off by 1;
  - price up to 10% over the max, or up to 10% under the min;
  - size up to 10% under the minimum;
  - one of two requested amenities missing.
- **1:** the area and property type hold (where stated), but the conditions
  for grades 2 and 3 fail.
- **0:** anything else. A wrong area is never a near miss: DLD has no
  coordinates, so "nearby" is undefined.
- **Amenities** are matched as case-insensitive whole phrases in
  `description`.
- **Fraud cap:** listings with a non-null ground-truth `fraud_label` are
  capped at grade 1.
- **What gets graded:** only the retrieved candidates of each query are
  stored with grades, in `search.judgments` (`query_id`, `listing_id`,
  `grade`). Grade-3 listings that retrieval missed are counted separately, for
  retrieval recall only.

## Parser (parse.py, lexicon.py)

- `ParsedQuery` fields: `area_ids: tuple[int, ...]`, `building`, `project`,
  `bedrooms`, `property_type`, `budget_min`, `budget_max`, `min_size_sqm`,
  `amenities: tuple[str, ...]`, `free_text`, `unrecognised: tuple[str, ...]`,
  and `errors: tuple[str, ...]`.
- **Lexicon:** official area names, `dld.area_aliases`, and the distinct
  building and project names in the corpus. Keys are normalised with the same
  function Phase 2 uses. Matching is greedy, longest phrase first, and the
  lexicon is loaded once per process.
- **Studio versus type:** "studio" sets `bedrooms=0` only. It does not set a
  type.
- **Money:** "under / below / max / up to / less than X", "X–Y", "between X
  and Y", "from X". The suffixes k, K, m, M, mn, million and "AED" are
  optional, as are thousands separators.
- **Bedrooms:** "studio" maps to 0, plus "N BR", "N bed(room)(s)", "N-bed",
  and the number words one to seven.
- **Size:** "over/at least N sqm|sq m|m2|sqft|sq ft". Square feet are
  converted with 0.092903.
- **Type:** matched against the corpus's four listing kinds. The kinds are
  derived the same way as `listings.generate._sub_kind`, from `property_type`
  (`unit`/`villa`) and `property_sub_type`.
  - "apartment" or "flat" maps to `flat`.
  - "hotel apartment" maps to `hotel_apartment`.
  - "townhouse" maps to `townhouse`.
  - "villa" maps to `villa`.
  - "plot" or "land" is not in the corpus. It goes to `unrecognised`, with the
    note "property type not listed".
  - Retrieval filters on the derived kind (a SQL `CASE` matching
    `_sub_kind`), and `true_slots` stores the kind.
- **Amenities:** matched against `AMENITIES`.
- **Free text:** whatever is left after slot phrases are removed.
- **Errors:** budget min > max produces
  `errors=("budget_min_exceeds_max",)`, and both bounds are dropped. A place
  word that isn't in the lexicon (after "in"/"at"/"near") goes to
  `unrecognised`.
- **Evaluation:** per-slot exact-match accuracy against `true_slots` on the
  report split, logged as `parse.<slot>.accuracy`.

## Retrieval (retrieve.py)

- **Semantic channel:** MiniLM embeds the full query text. pgvector kNN
  (`<=>`) runs on `text_embedding`, with top `semantic_k = 200`.
  `hnsw.ef_search = 400`, and `hnsw.iterative_scan = relaxed_order` so the
  filters don't starve the result set.
- **Full-text channel:** a generated column
  `listings.listings.search_tsv tsvector` over `title || ' ' || description`,
  using the `english` configuration, with a GIN index. The column is added by
  `search/sql/schema.sql` with `ADD COLUMN IF NOT EXISTS`. The query is
  `websearch_to_tsquery` over `free_text` plus the amenity phrases, falling
  back to the full text when those are empty. Ranking uses `ts_rank_cd`, with
  top `fulltext_k = 200`.
- **Filters on both channels:** `area_id = ANY(parsed.area_ids)` and
  the derived kind `= parsed.property_type`, applied only when the parser
  produced them. Bedrooms, budget and size are *not* filtered.
- **Fusion:** reciprocal rank fusion with `rrf_k = 60`, keeping
  `candidate_k = 200`.
- **Duplicate collapse:**
  - Clusters are the connected components of the latest detect run's
    `duplicate_pairs` where `decision` is true.
  - They are computed once per process and cached.
  - Each cluster keeps its earliest-posted member; ties go to the lowest
    `listing_id`.
  - `cluster_size` is carried as a feature.
  - Collapse happens before truncation to `candidate_k`.
- **Fallbacks:**
  - A query with no parsed slots and no free text returns an empty result
    with reason `empty_query`.
  - A query with no filters runs both channels unfiltered.
- **Metric:** `retrieval.recall_at_200` is the share of a query's grade-3
  listings (computed over the whole corpus by `grade`) that appear among its
  candidates, averaged over report queries that have at least one grade-3
  listing.

## Features (features.py)

`FEATURES` is an ordered tuple. It is computed from `ParsedQuery` and never
from `true_slots`.

- **Match features:**
  - `beds_diff`: abs(listing − query), or NaN when not stated;
  - `beds_stated`;
  - `price_over_max`: price ÷ budget_max − 1, or NaN;
  - `price_under_min`;
  - `budget_stated`;
  - `size_ratio`: size ÷ min_size, or NaN;
  - `area_match` and `area_stated`;
  - `type_match` and `type_stated`;
  - `building_match`;
  - `amenity_hits`, `amenity_asked`.
- **Similarity features:**
  - `semantic_cos`;
  - `fulltext_rank`, 0 when the listing was not retrieved by that channel;
  - `semantic_pos`, `fulltext_pos`, the position in each channel (NaN when
    absent);
  - `rrf_score`.
- **Value features:**
  - `price_to_estimate`: asking price ÷ the Phase 3 estimate;
  - `within_interval`: whether the asking price falls inside the 80% range.
  - Estimates are computed once for all listings and cached in
    `search.listing_estimates` (`listing_id`, `estimate`, `low`, `high`,
    `price_model_version`).
  - If the price model is unavailable, both features are NaN and
    `stats.value_features_skipped = 1`.
- **Trust features:**
  - `flag_bait_price`, `flag_photo_reuse`, `flag_inconsistent_relist`: the
    predicted flags from `listings.fraud_flags`;
  - `cluster_size`.
- **Freshness:** `days_since_posted`, measured against the corpus's latest
  `posted_at`, so results are reproducible.
- **Leakage guard:** a test scans `search/` (excluding `grade.py`,
  `evaluate.py` and `queries.py`) for `fraud_label`, `dup_group_id`,
  `control_group_id`, `true_slots` and `is_synthetic`, in SQL and in code.

## Ranker (train.py)

- **Data:** one row per (query, candidate), grouped by query.
  - Train split: fit.
  - Tune split: Optuna objective and early stopping.
  - Report split: never touched until evaluation.
- **Champion candidate:** XGBoost `objective="rank:ndcg"`, `device="cuda"`,
  `eval_metric="ndcg@10"`, `lambdarank_pair_method="topk"`.
- **Challenger:** LightGBM `objective="lambdarank"`, `metric="ndcg"`,
  `eval_at=[10]`, on the CPU. It relies on the `models.price` DLL preload
  already in place.
- **Tuning:** Optuna with `n_trials = 40` per model and a TPE sampler seeded
  from `--seed`. The search covers depth, learning rate, number of trees
  (with early stopping, 50 rounds), min child weight, subsample and column
  sample.
- **Refit:** the final model is refit on train only, with the best
  parameters and best iteration count. The report split is never used for
  fitting, tuning or model selection. Only the registration gate reads it,
  once, after the winner is fixed.
- **Winner:** the model with the higher tune NDCG@10.
- **Registration gate:** the winner is registered as
  `zestimator-search-ranker` with alias `@champion` only if both hold:
  1. its report NDCG@10 exceeds the fused-retrieval baseline's;
  2. the lower bound of its bootstrap 95% CI exceeds that baseline's point
     estimate.

  Otherwise the run logs `gate.passed = 0` and registers nothing.
- **Model format:** an MLflow pyfunc wrapping the booster and `FEATURES`. The
  model signature checks the feature order.

## Evaluation (evaluate.py)

All metrics are computed on the **report** split, unless prefixed `tune.`.

- **Ranking metrics for every contender:**
  - `ndcg_at_10`, using gains 2^g − 1;
  - `mrr`: first result with grade ≥ 2;
  - `precision_at_5`: grade 3.
- **Contenders:**
  - `xgb`;
  - `lgbm`;
  - `baseline_newest`: filtered candidates, newest first;
  - `baseline_semantic`: semantic channel order;
  - `baseline_fused`: RRF order.
- **Bootstrap:** 1,000 resamples over queries with a fixed seed. For each
  contender the run logs `.ci_low` and `.ci_high`.
- **By kind:** `ndcg_at_10` per query kind for every contender. For
  `no_match`, the metric is the mean top-10 grade, logged as
  `kind.no_match.mean_grade_top10`.
- **Retrieval and parser:** `retrieval.recall_at_200` and
  `parse.<slot>.accuracy`.
- **Phase 4 effects:**
  - `dup.top10_removed`: the mean number of duplicates the collapse removed
    from the top 10;
  - `fraud.top10_share`: the share of ground-truth fraud listings in the
    champion's top 10;
  - the same share for an ablation ranker trained without the trust features
    (same parameters, logged as `ablation.no_trust.*`).
- **Undefined values** are NaN. `log_run` drops non-finite values, following
  Phase 4.
- **Artifacts:**
  - `ndcg_comparison.png`: bars with CI whiskers;
  - `feature_importance.csv`;
  - `per_query_report.csv`: query text, kind, parsed slots, NDCG for each
    contender;
  - `parse_errors.csv`: the first 200 slot mismatches;
  - `timings.json`.
- **MLflow:** experiment `search-ranking`. Params record the seed, the
  counts, `price_model_version` and the detect run id used.

## Engine (engine.py)

- **Signature:** `search(conn, text: str, k: int = 10) -> SearchResult`.
- **`SearchResult` fields:** `parsed: ParsedQuery`, `results: list[Hit]`,
  `ranker: str` (a model version, or `"fallback_fused"`), `notes: list[str]`,
  and `timings_ms: dict`.
- **`Hit` fields:** `listing_id`, `score`, `title`, `area_name`, `bedrooms`,
  `asking_price_aed`, `size_sqm`, `reasons: list[str]`, and
  `duplicates_hidden: int`.
- **Reasons** are built from the match and trust features. Examples:
  "2 bedrooms ✓", "4% over budget", "area ✓", "sea view ✓", "priced 12%
  below estimate", "flagged: bait price".
- **Ranker loading:** the ranker is loaded once, lazily, from
  `models:/zestimator-search-ranker@champion`. If it is missing or fails to
  load, the engine falls back to fused order, sets
  `ranker="fallback_fused"`, adds a note, and logs one warning naming the
  tracking URI.
- **Notes also cover:**
  - `unrecognised` places ("area not recognised: …");
  - parser `errors`;
  - an empty result ("no listings match; try widening the budget").

## CLI (`python -m search`)

- `queries [--n 6000] [--seed 7]`: generate queries, run retrieval, grade the
  candidates, and write `search.queries` and `search.judgments`. Also fills
  `search.listing_estimates` if it is empty.
- `train [--trials 40]`: fit both rankers, select the winner, and register it
  if the gate passes.
- `evaluate`: score every contender and log to MLflow. This may be merged
  into `train`'s run, per the plan.
- `query "<text>" [--k 10]`: print the parsed slots and the ranked hits with
  their reasons.

Behaviour shared with Phases 2–4:
- `load_dotenv()` and `logging.basicConfig` run before dispatch;
- `DbSettings.from_env()` refuses an unset port;
- each stage writes its wall-clock time to
  `data/search/stage_timings.json`, which is gitignored.

## Storage (search/sql/schema.sql)

- Schema `search`, with three tables:
  - `queries`: primary key `query_id`, with a CHECK on `split` and `kind`;
  - `judgments`: primary key `(query_id, listing_id)`, grade CHECK 0–3;
  - `listing_estimates`: primary key `listing_id`.
- `listings.listings.search_tsv`: a generated stored column, plus the GIN
  index `listings_search_tsv_idx`.
- The `queries` stage rewrites `search.*` atomically (truncate and COPY in one
  transaction), as Phase 4 does. It never touches the Phase 4 tables other
  than adding the generated column and index.

## Testing

Tests run sequentially only, against `zestimator_test`.

- **Parser:** a table of about 60 cases covering every slot form, alias
  matching, longest match, "studio", ranges, sq ft conversion, min > max
  error, unrecognised place, and empty string.
- **Grading:** hand-built cases for each grade boundary, including the fraud
  cap and the rule that a wrong area is never grade 2.
- **Queries:**
  - determinism under the seed;
  - kind shares within ±3 points;
  - `no_match` queries truly have no grade-3 listing;
  - no template or exact text is shared across splits;
  - every split contains every kind.
- **Retrieval, on the test DB with `FakeEmbedder` and a small corpus:**
  - both channels return results;
  - filters are applied;
  - RRF order is hand-checked;
  - collapse keeps the earliest member and reports `cluster_size`.
- **Features:** arithmetic hand-checked, NaN when not stated, and value
  features NaN with a stub-less (missing) price model.
- **Leakage scan** as described above.
- **Ranker:** fits on a small fixture on the CPU (the test forces `device`),
  shows NDCG improving over fused order on a constructed easy fixture, and
  enforces the gate both ways.
- **Evaluation:** metric values hand-checked on 2–3 toy queries, bootstrap
  determinism, and NaN handling.
- **Engine:** results with a registered champion (temporary MLflow), the
  fallback when none is registered, reasons text, and notes for unrecognised
  places and errors.
- **CLI:** an end-to-end run on the test DB. The test asserts
  `DbSettings.from_env().dbname == "zestimator_test"` before running.
- **Lint and warnings:** `ruff check`, `ruff format --check`, and
  `pytest -W error`.

## Deliverables beyond code

- **Real run:** 6,000 queries on the 20,000-listing corpus, with the GPU used
  for the XGBoost ranker and query embedding. Timings are recorded per stage.
- **README:** a "## Property search" section that includes:
  - which data is synthetic;
  - the report-split table of contenders with CIs, the per-kind breakdown,
    parser accuracy and retrieval recall;
  - the Phase 4 effects (duplicates removed, fraud share, and the ablation);
  - the in-sample caveat on the value features;
  - known limitations.

  Every number must come from the run's logs or its MLflow run.
- **Architecture page:** republish `docs/architecture.html` to the same
  artifact URL.
- **Spec amendments:** changes made during implementation are recorded at the
  bottom of this file.

## Out of scope

- Real user queries, click logs, or online A/B testing.
- Geographic proximity (no coordinates in DLD data).
- Personalisation, pagination, query autocomplete, spelling correction.
- Serving over HTTP. That is Phase 6, which will call `search.engine.search`.
- Re-embedding listings or changing any Phase 4 detection logic.

## Amendments during implementation (2026-09-16)

### Planning rulings (from the plan header)

1. **Judgments store the retrieval signals.** `search.judgments` carries
   `semantic_cos`, `semantic_pos`, `fulltext_rank`, `fulltext_pos`,
   `rrf_score`, `fused_pos` and `cluster_size` next to `grade`. Training and
   evaluation therefore never re-run retrieval for 6,000 queries.
2. **Duplicate-collapse representative.** The representative is the
   earliest-posted member *among the retrieved members* of a cluster (ties go
   to the lowest `listing_id`), not the earliest member overall. Its retrieval
   signals and features are its own, never borrowed from a sibling.
3. **`ParsedQuery.unrecognised` is a tuple of `(kind, text)` pairs**, with
   kind `"place"` or `"type"`. The engine can then say "area not recognised:
   …" or "property type not listed: land".
4. **Corpus pinning.**
   - `search.queries` and `search.listing_estimates` carry `corpus_run_id`.
   - `train` and `evaluate` refuse to run when the stored queries were built
     on an older corpus. The error tells the user to run
     `python -m search queries`.
   - `queries` recomputes the estimates whenever they belong to an older
     corpus, not only when the table is empty.
6. **`search/engine.py` exposes a `SearchEngine` class plus the module-level
   `search(conn, text, k=10)`.**
   - The class caches the lexicon, clusters, attributes, flags, estimates and
     ranker.
   - Its docstring states the contract: one long-lived engine per worker,
     holding one connection, not safe for concurrent use from several
     threads.
   - The module-level `search` caches one engine per connection object and
     never evicts it. The pool-aware engine is left to the API phase (now
     Phase 7).
7. **`train` fits, evaluates, gates and logs in one MLflow run
   (`search-train`).**
   - `evaluate` re-scores the registered champion and the three baselines in
     a separate `search-evaluate` run, so results can be reproduced without
     refitting.
   - The Phase 4 effects and the no-trust ablation are logged only by
     `train`, which has both fitted models in hand.
8. **The 30 templates are frames: 6 prefixes × 5 slot orders.**
   - A frame id also picks the phrasing style for bedrooms and budget.
   - The split assigns 18 frames to train, 6 to tune and 6 to report.
   - Per-query randomness (alias or official name, number format) is
     independent of the frame.

### Parser

- **Pass order.** The parser runs its passes in this order:
  1. size;
  2. budget;
  3. **places**;
  4. bedrooms;
  5. property type;
  6. amenities;
  7. unrecognised places.

  Places now run *before* bedrooms and type because real DLD area, building
  and project names contain slot words: "studio", "villa", number words and
  digits. Matching the lexicon first blanks those names out, so their words
  are not read as bedroom counts or property types.
- **`ParsedQuery` fields.**
  - It gained `area_name`, the matched area as the lexicon spells it.
  - It has no separate `project` field: `building` holds a building *or*
    project name, and the grade rule accepts either.
- **Extra input forms**, added after review:
  - "N+ bed";
  - en and em dash ranges;
  - range-suffix inheritance. "1-1.5M" inherits the "M". "900-1.2M" falls
    back to thousands when the inherited first bound would exceed the
    second. A first number that is already an amount ("1,500-2,000k")
    inherits nothing.
- **Known limitation, accepted.** A possessive ("Marina's Gate") breaks the
  building match.

### Leakage scan

- **`store.py` is exempt from the label-name scan.** It is listed in
  `LABEL_READERS` next to `grade.py`, `queries.py` and `evaluate.py`, and
  `config.py` is exempt because it declares the forbidden list.
- **Why the exemption is needed.** `store.py` persists `true_slots` with the
  query set, but its `read_queries()` leaves them out.
- **The scan covers everything else** in `search/`, including `engine.py` and
  `__main__.py`.

### Metrics and evaluation

- **Contender names.** The contenders are logged as `xgboost` and `lightgbm`,
  not `xgb` and `lgbm`.
- **Which queries count.** NDCG@10, MRR, P@5 and the bootstrap CI average over
  *answerable* queries: kind other than `no_match`, with at least one
  candidate graded ≥ 1. `no_match` queries are scored only by
  `kind.no_match.<contender>.mean_grade_top10`.
- **Score ties and gaps.** Candidates with a NaN score are ranked last (ties
  keep fused order), and `precision_at_k(k=0)` is NaN.
- **Recall denominator.** `retrieval.recall_at_200` divides by *all* of a
  query's grade-3 listings in the corpus (`search.queries.n_grade3`),
  uncapped. It is therefore bounded by `candidate_k` on broad queries (see
  the headline numbers below).

### CLI

- **Flags beyond the spec.**
  - `queries`: `--device auto|cuda|cpu`, `--fake` (deterministic test
    embedder, tests only) and `--data-dir`.
  - `train`: `--device` and `--data-dir`.
  - `query`: `--fake` and `--device`.
- **Stats file.** `queries` writes `queries_stats.json`: the estimate stats
  plus per-split and per-kind counts. It prints a stderr WARNING if the
  price model could not be loaded and the value features were skipped.
- **Gate failure.** A failing gate is a legitimate result: `train` still
  exits 0 and registers nothing, and `evaluate` then exits 1 with "no
  champion registered".
- **Console encoding.** The CLI switches a non-UTF-8 stdout or stderr to
  UTF-8 (commit 8d4b5e6). The first real run found that a piped Windows
  console (cp1252) could not encode the "✓" in the reasons, so `query`
  failed.

### Other

- **Test fixture.** The `temp_mlflow` fixture touches the sqlite store during
  setup. MLflow's alembic migration replaces the root logging handlers, and
  running it at setup keeps pytest's `caplog` working in the test body.
- **Phase numbering.** Phases were renumbered on 2026-09-16, when price
  forecasting became Phase 6. Where this spec says "Phase 6" for serving over
  HTTP, read Phase 7 (API).
- **Ranker label.** The engine labels its ranker
  `zestimator-search-ranker/v<version>`, or `fallback_fused`.

### Real run and headline numbers (2026-09-16, RTX 3060)

**Stages** (wall-clock):
- `queries`: 1,023s. It embedded on cuda, used the Phase 3 champion
  (version 2) and wrote 19,983 estimates, with 17 listings unsupported. It
  produced 6,000 queries and 1,098,438 graded candidates. The report split
  holds 864 `specified`, 215 `vague` and 113 `no_match` queries.
- `train`: 1,454s, with 40 trials per model.
- `evaluate`: 22s.
- MLflow run `search-train`: `daddafc0afe940fc97bad7a58826ac80`.

**Results** (report split, 1,079 answerable queries):

| Contender | NDCG@10 (95% CI) |
|---|---|
| LightGBM (winner) | 0.997 (0.995–0.999) |
| XGBoost | 0.997 (0.995–0.998) |
| Fused retrieval (`baseline_fused`) | 0.566 (0.549–0.583) |
| Semantic channel | 0.470 |
| Newest first | 0.435 |
| No-trust ablation | 0.997 |

- **Winner and gate.** Tune NDCG@10 was 0.9980 for LightGBM against 0.9977
  for XGBoost. The gate passed, and the winner was registered as
  `zestimator-search-ranker` v1 with alias `@champion`.
- **By kind.** `specified` NDCG@10 is 0.996 against 0.483 for fused
  retrieval. `vague` is 1.000 against 0.897. The `no_match` mean top-10
  grade is 1.00 for every contender.
- **Retrieval recall@200** is 0.525.
- **Parser accuracy** is 1.000 on every slot except `building`, at 0.996.
- **Phase 4 effects.**
  - `dup.top10_removed`: 0.45 (0.41 for the ablation).
  - `fraud.top10_share`: 0.88% (0.93% for the ablation).
- **Interpretation.** The grades are rule-based and the features restate the
  graded slots, so the near-perfect ranker NDCG mainly shows the ranker
  reproduces the grading rules. The README states this plainly.
- **Phase 4 tables unchanged.** The run changed no Phase 4 table beyond
  adding `listings.listings.search_tsv` and its index. Row counts and
  content hashes of every other `listings` table were identical before and
  after.

## Amendments after the final review (2026-09-16)

The first run's numbers above are kept as history. The numbers in this section
supersede them.

### Rulings

- **A14: re-run after parser changes.** The parser fixes below change parsing,
  so the published numbers come from a fresh `queries`, `train` and
  `evaluate` run on the fixed code.
- **A15: gate against the stronger baseline.** The registration gate now
  compares the winner with the stronger of `baseline_fused` and the new
  `baseline_rules`. Both conditions still apply: the winner's report NDCG@10
  must exceed that baseline's NDCG@10, and so must the lower bound of the
  winner's 95% CI.
  - `train` logs the baseline used as param `gate.baseline`, and its NDCG@10
    as metric `gate.baseline_ndcg`.
- **A16: model packaging.** `log_ranker` no longer passes `code_paths`, so
  MLflow no longer prepends a bundled copy of `search/` to `sys.path`.
  - The ranker is only ever loaded in-process from this repository, which
    must be importable.
  - There is **no model signature**, which supersedes "the model signature
    checks the feature order" under Ranker. The ranker selects its features
    from the input frame by name, in `FEATURES` order.

### Parser

- **One-word building and project names** (`Place.single_token`) match only
  right after "in", "at", "near" or "around", optionally followed by "the".
  One-word derived community names follow the same rule.
- **A building never filters by area.** `ParsedQuery.building_area_ids`
  holds a named building's areas. `area_ids` holds only directly named areas,
  so it is the only area filter in retrieval and the only area checked by
  the reasons.
- **Community names.** `load_lexicon` reads `master_project` aliases
  separately, and each one yields derived names:
  - the part before the first " - ";
  - a base name without a trailing number or roman numeral, also dropping a
    "Phase" that precedes it;
  - both spellings of the number ("II" and "2");
  - each name without a leading "The".

  Each derived name maps to the union of its aliases' areas. It never
  replaces an official, curated or existing alias key, but like every area
  key it wins over a building or project name. On the real data, 144
  aliases yield 27 derived names. Examples: "Arabian Ranches" maps to areas
  434, 452 and 463, "Arabian Ranches 2" to 463, and "Springs" and "Meadows"
  to 352.
- **"in the X"** now notes X as an unrecognised place.

### Engine

- **Session settings.** `retrieve.configure_session` sets `hnsw.ef_search`
  and `hnsw.iterative_scan` with a plain, session-level `SET`.
  - The engine calls it once, then commits, because Postgres undoes a
    session `SET` made inside a transaction that is later rolled back.
  - `build_query_set` also calls it once.
  - `semantic_channel` issues no `SET`.
- **No open transactions.** The engine rolls back at the end of
  construction, and before and after the retrieval in every `search()`
  (also on error).
- **Warm-up.** Construction embeds "warm up" once, and `query` prints the
  construction time separately.
- **Stale estimates.** If the stored estimates do not belong to the latest
  corpus, or are missing, the engine ignores them (the value features become
  NaN) and adds the note "price estimates are out of date; value signals
  skipped".
- **Closest-match note.** "nothing meets every requirement; showing the
  closest matches" is added when no returned hit meets every parsed slot
  exactly (`features.meets_every_slot`).
- **Ordering and labels.** NaN scores sort last.
  - `ranker.load_pinned` resolves the alias to a version and then loads
    exactly that version, so the label and the model always agree.
  - `SearchResult` and `Hit` gain `to_dict()`.
- **Docstring.** It states the snapshot contract: there is no reload, and the
  engine must be rebuilt after `listings build`, `search queries` or a new
  champion. It also states that exceptions propagate, that
  `k ≤ candidate_k`, and that `search` must be imported first.

### Evaluation and leakage

- **`baseline_rules`.** It scores each candidate 3, 2 or 1 by applying the
  grading rules to the *parsed* slots (`features.rule_grade`) and breaks
  ties by `rrf_score`. It uses no labels.
  - A stated bedroom count against a listing with no bedroom count holds
    neither exactly nor nearly, as in `grade.py`.
  - The reviewer's probe counted that case as a match and got 0.960 on the
    first run's report split. This implementation gets 0.972 on the same
    split.
- **Per-contender Phase 4 effects.** Every contender, including the
  ablation, logs `<contender>.fraud.top10_share` and
  `<contender>.dup.top10_removed`.
  - The champion keys `fraud.top10_share` and `dup.top10_removed` remain.
  - `candidates.fraud_share` is the fraud base rate across report
    candidates.
  - For the champion, `note.closest.rate.no_match` and
    `note.closest.rate.other` give the closest-match note's rate by query
    kind.
- **Label readers.** `store.read_queries` no longer returns
  `seed_listing_id` or `n_grade3`, and `read_query_labels` (used only by
  `evaluate`) returns them. Both names are in `SEARCH_FORBIDDEN`, and the
  leakage scan exempts exactly `search/<name>.py`.
- **Timings.** `train` logs only this query set's `queries` timing plus
  `train_before_logging`.
- **Reading the metrics.** NDCG@10 measures re-ranking *within* the
  retrieved candidates. `n_grade3`, the recall denominator, also counts
  duplicate-cluster members that the collapse never shows.
- **What the template split holds out.** Each report frame is an unseen
  *combination*. Every opening phrase, slot order and bedroom/budget style
  in the report frames also occurs in some training frame.

### Re-run and headline numbers (2026-09-16, RTX 3060)

**Stages** (wall-clock):
- `queries`: 685s. It embedded on cuda and printed no skip warning. The
  estimates were already current (`estimates_cached` 1). It produced 6,000
  queries and 1,098,524 graded candidates. The report split again holds 864
  `specified`, 215 `vague` and 113 `no_match` queries.
- `train`: 1,177s.
- `evaluate`: 18s.

**Runs:**
- MLflow `search-train`: `6682999844ed46adad2a4720b1f1d619`.
- `search-evaluate`: `549171a2a35b45fab5f0e003de5cd958`, which reports
  ranker version 2.

**Results** (report split, 1,079 answerable queries):

| Contender | NDCG@10 (95% CI) | Top-10 fraud share |
|---|---|---|
| LightGBM (winner) | 0.997 (0.995–0.999) | 0.96% |
| XGBoost | 0.997 (0.995–0.998) | 0.93% |
| Rules over parsed slots (`baseline_rules`) | 0.972 (0.968–0.976) | 3.95% |
| Fused retrieval (`baseline_fused`) | 0.566 (0.549–0.583) | 3.13% |
| Semantic channel | 0.470 (0.454–0.486) | 3.22% |
| Newest first | 0.435 (0.419–0.450) | 4.10% |
| No-trust ablation | 0.997 (0.995–0.998) | 0.98% |

- **Base rate.** The fraud base rate across report candidates is 3.45%.
- **Lifts** (paired bootstrap over queries):
  - learned over rules: +0.025 (0.021–0.029);
  - learned over fused: +0.431 (0.414–0.449).
- **Gate: passed against `baseline_rules`** (0.972). LightGBM was
  registered as `zestimator-search-ranker` v2 with alias `@champion`. Tune
  NDCG@10 was 0.9979 for LightGBM against 0.9977 for XGBoost.
- **By kind:**
  - `specified`: 0.996 for the winner, 0.975 for the rules and 0.483 for
    fused;
  - `vague`: 1.000, 0.963 and 0.897;
  - `no_match`: the mean top-10 grade is 1.00 for all seven contenders.
- **Closest-match note** (winner's top 10): it would appear on 100% of the
  `no_match` queries and on 1.9% of the others.
- **Retrieval and parsing.** Retrieval recall@200 is 0.525 (0.646 with the
  denominator capped at 200). Parser accuracy is 1.000 on every slot except
  `building`, at 0.996, with the same five misses as the first run.
- **Unchanged report figures.** The report-split fused, recall and parser
  figures are identical to the first run's.
- **Duplicates.** `dup.top10_removed` is 0.44 for the winner and 0.40 for
  the ablation.
