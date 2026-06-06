# Documentation and Writing Rules

This guide applies to all prose in this repository, except text copied verbatim from reference literature.

Reference literature has higher priority than these rules. If a sentence, phrase, term, or wording is copied from a cited source, keep it as written by the original author, even if it violates this guide.

For example, if Boyd writes:

> “The optimization problem (1.1) is an abstraction of the problem of making the best possible choice of a vector in $\mathbb{R}^n$ from a set of candidate choices.”

then that wording may be used as-is. Do not rewrite it only to satisfy this guide.

---

These rules apply to original repository text, including:

* Markdown sections in notebooks
* Code comments
* Docstrings
* `README.md` files
* Other Markdown documentation
* Agent-facing instructions, including** **`AGENTS.md`,** **`*-RULES.md`, and prompts
* Commit messages and PR descriptions
* Any other** **`.md` file

Use plain, factual, technical writing.

Avoid:

* Marketing tone
* Filler
* AI-sounding boilerplate
* Vague claims
* Unnecessary explanation

Read this file before writing or editing repository documentation.

## 1. Core Principles

1. **Write for an engineer who needs to act on the text.** Not for a reader you
   want to impress.
2. **Say what the thing is and how it works.** Skip context that the reader
   already has from the surrounding file or repo.
3. **Be specific.** Prefer concrete nouns, file paths, function names, units,
   and numbers over abstractions.
4. **One claim per sentence.** Short sentences. No nested clauses unless
   needed.
5. **Cut anything that does not change the reader's behavior or
   understanding.** If removing a sentence loses no information, remove it.
6. **Match the existing tone of the file.** Do not "improve" surrounding text
   while editing.

---

## 2. Words and Phrases to Avoid

These words appear constantly in AI-generated text and rarely add meaning.
Replace them or delete the sentence.

### 2.1 Inflated verbs

Here is a cleaned and expanded version:

| Avoid                        | Use instead                                                               |
| ---------------------------- | ------------------------------------------------------------------------- |
| leverage                     | use                                                                       |
| utilize / utilise            | use                                                                       |
| facilitate                   | let, allow, help, run                                                     |
| enable                       | let, allow; use only when technically exact                               |
| empower                      | let, allow                                                                |
| streamline                   | simplify, remove, shorten                                                 |
| optimize                     | improve, speed up, reduce memory, reduce latency; use only when measuring |
| enhance                      | improve, extend, add                                                      |
| elevate                      | improve, raise                                                            |
| foster                       | encourage, cause                                                          |
| harness                      | use                                                                       |
| unlock                       | expose, give access to, allow                                             |
| unleash                      | release, run, start                                                       |
| drive                        | cause, run, lead to                                                       |
| craft                        | write, build, make                                                        |
| delve into / dive into       | look at, read, inspect, study                                             |
| explore                      | look at, test, try                                                        |
| embark on                    | start                                                                     |
| navigate                     | handle, walk through, work with                                           |
| showcase                     | show                                                                      |
| underscore / underline       | show, prove, mean                                                         |
| shed light on                | explain, show                                                             |
| shape                        | change, define                                                            |
| transform                    | change, replace, rewrite                                                  |
| revolutionize                | delete                                                                    |
| reimagine                    | redesign, rewrite                                                         |
| facade / façade             | wrapper, interface, shell, front; keep only for the Facade pattern        |
| robust                       | reliable, strict, fault-tolerant; or describe the actual behavior         |
| seamless                     | simple, automatic, without manual steps                                   |
| scalable                     | works for N, supports larger data, handles more users                     |
| comprehensive                | complete, full, covers X                                                  |
| holistic                     | complete, end-to-end, broad                                               |
| intuitive                    | clear, easy to read, easy to use                                          |
| user-friendly                | easy to use, clear                                                        |
| cutting-edge                 | new, recent; or delete                                                    |
| state-of-the-art             | current best known, modern; or delete                                     |
| game-changing                | important, large, useful; or delete                                       |
| innovative                   | new, different; or delete                                                 |
| impactful                    | useful, important, measurable                                             |
| powerful                     | useful, flexible, fast                                                    |
| dynamic                      | changing, runtime, configurable                                           |
| tailored                     | custom, specific to X                                                     |
| bespoke                      | custom                                                                    |
| curated                      | selected                                                                  |
| rich                         | detailed, large, structured                                               |
| deep                         | detailed, low-level, nested                                               |
| smart                        | automatic, rule-based, model-based                                        |
| intelligent                  | automatic, model-based                                                    |
| magical / magic              | implicit, hidden, automatic                                               |
| frictionless                 | simple, fewer steps                                                       |
| effortlessly                 | easily, with one step, automatically                                      |
| seamlessly integrates        | connects to, works with                                                   |
| integrate                    | connect, add, wire                                                        |
| orchestrate                  | coordinate, run in order                                                  |
| ecosystem                    | project, repo, tools, packages                                            |
| landscape                    | area, field, codebase                                                     |
| realm                        | area, module, layer                                                       |
| journey                      | process, steps                                                            |
| tapestry                     | system, structure; usually delete                                         |
| paradigm                     | pattern, model, approach                                                  |
| synergy                      | interaction, combined effect; usually delete                              |
| alignment                    | agreement, match, consistency                                             |
| cohesive                     | consistent, connected                                                     |
| nuanced                      | specific, detailed                                                        |
| pivotal                      | important                                                                 |
| paramount                    | important                                                                 |
| critical                     | required, blocking, important                                             |
| essential                    | required, needed                                                          |
| vital                        | required, important                                                       |
| sophisticated                | complex, advanced                                                         |
| elegant                      | simple, clean                                                             |
| meticulous                   | careful, exact                                                            |
| thoughtfully                 | carefully; usually delete                                                 |
| carefully crafted            | written, built                                                            |
| strategically                | intentionally; usually delete                                             |
| seamlessly handle            | handle                                                                    |
| gracefully handle            | handle safely, return X, catch X                                          |
| gracefully degrade           | fall back to X                                                            |
| future-proof                 | easier to change, less coupled                                            |
| production-ready             | tested, typed, documented, deployable                                     |
| enterprise-grade             | reliable, audited, configurable; or delete                                |
| battle-tested                | used in production, tested under X                                        |
| bulletproof                  | safer, guarded, validated                                                 |
| blazing-fast                 | fast, O(n), X ms, cached                                                  |
| lightning-fast               | fast, X ms                                                                |
| performant                   | fast, efficient                                                           |
| clean architecture           | separated layers, low coupling                                            |
| clean code                   | readable code, smaller functions                                          |
| best practice                | common pattern, project rule, required convention                         |
| industry-standard            | common, widely used                                                       |
| canonical                    | standard, default                                                         |
| idiomatic                    | typical for X language/framework                                          |
| modernize                    | update, replace old X with Y                                              |
| revamp                       | rewrite, restructure                                                      |
| overhaul                     | rewrite, replace major parts                                              |
| improve developer experience | make setup easier, reduce steps, improve errors                           |
| DX                           | developer experience; only use if project uses this term                  |
| surface                      | expose, show, return                                                      |
| ingest                       | read, load, import                                                        |
| emit                         | return, write, send                                                       |
| consume                      | read, call, use                                                           |
| hydrate                      | load full data, fill object                                               |
| persist                      | save, write to disk/db                                                    |
| bootstrap                    | initialize, create initial setup                                          |
| scaffold                     | create starter files                                                      |
| plug-and-play                | works without extra config                                                |
| drop-in replacement          | replacement with same API                                                 |
| single source of truth       | canonical config, shared value, one owner                                 |
| source of truth              | canonical value, primary record                                           |
| guardrails                   | checks, limits, validation                                                |
| safe by default              | default rejects X, default validates X                                    |
| opinionated                  | strict, project-specific                                                  |
| declarative                  | config-based, describes state                                             |
| imperative                   | step-by-step                                                              |
| ergonomic                    | easier to use, simpler API                                                |
| extensible                   | easy to add X                                                             |
| modular                      | split by module, separated                                                |
| decoupled                    | less dependent on X                                                       |
| loosely coupled              | depends only on interface X                                               |
| abstract away                | hide, wrap                                                                |
| abstraction                  | wrapper, interface, base class                                            |
| layer                        | module, package, boundary                                                 |
| pipeline                     | steps, flow; keep if it is really a pipeline                              |
| workflow                     | steps, process                                                            |
| flow                         | path, steps, control flow                                                 |
| bridge                       | connect, adapter                                                          |
| align with                   | match, follow                                                             |
| conform to                   | follow                                                                    |
| comply with                  | follow; use only for rules/law/security                                   |
| adhere to                    | follow                                                                    |
| ensure                       | make sure, check, enforce                                                 |
| guarantee                    | ensure only if mathematically/technically guaranteed                      |
| validate                     | check                                                                     |
| verify                       | check, prove, test                                                        |
| establish                    | create, define, set                                                       |
| provide                      | give, expose, return                                                      |
| support                      | allow, accept, handle                                                     |
| address                      | fix, handle                                                               |
| tackle                       | fix, handle                                                               |
| mitigate                     | reduce, avoid                                                             |
| resolve                      | fix                                                                       |
| remediate                    | fix                                                                       |
| augment                      | add to, extend                                                            |
| iterate                      | repeat, revise                                                            |
| iterate on                   | revise, update                                                            |
| refine                       | clean up, improve                                                         |
| polish                       | clean up                                                                  |
| solidify                     | finalize, make stable                                                     |
| clarify                      | explain, rename, document                                                 |
| articulate                   | explain, write                                                            |
| encapsulate                  | wrap, contain; keep if OOP meaning is exact                               |
| compose                      | combine                                                                   |
| consolidate                  | merge                                                                     |
| centralize                   | move to one place                                                         |
| decentralize                 | split across modules                                                      |
| parameterize                 | make configurable                                                         |
| standardize                  | make consistent                                                           |
| normalize                    | make consistent; keep if data/math meaning is exact                       |
| canonicalize                 | convert to standard form                                                  |
| serialize                    | encode/write; keep if exact                                               |
| deserialize                  | parse/read; keep if exact                                                 |
| materialize                  | create, compute, write                                                    |
| derive                       | compute from X                                                            |
| infer                        | guess, compute, determine                                                 |
| inferencing                  | inference                                                                 |
| enrich                       | add fields/data                                                           |
| sanitize                     | clean, escape, validate                                                   |
| harden                       | add checks, make safer                                                    |
| resilience                   | recovery behavior, retry behavior                                         |
| observability                | logs, metrics, traces                                                     |
| telemetry                    | metrics/events                                                            |
| instrumentation              | logging/metrics code                                                      |
| seamless experience          | simple path, fewer steps                                                  |
| rich experience              | useful UI, detailed UI                                                    |
| delightful                   | clear, fast, simple; usually delete                                       |
| captivating                  | delete                                                                    |
| immersive                    | delete unless UI/media context                                            |
| compelling                   | clear, strong, convincing                                                 |
| thoughtfully designed        | designed                                                                  |
| beautifully                  | delete                                                                    |
| unlock the power of          | use                                                                       |
| in today’s fast-paced world | delete                                                                    |
| ever-evolving                | changing                                                                  |
| at your fingertips           | available                                                                 |
| take it to the next level    | improve                                                                   |
| next-generation              | new, updated; or delete                                                   |
| world-class                  | high-quality; or delete                                                   |
| mission-critical             | required for X                                                            |
| end-to-end                   | full path from X to Y                                                     |
| at scale                     | with N users/records/requests                                             |
| real-time                    | live, streaming; only if technically true                                 |
| near-real-time               | delayed by X seconds/minutes                                              |
| actionable insights          | useful findings, metrics, results                                         |
| insights                     | findings, metrics, results                                                |
| visibility                   | logs, metrics, status                                                     |
| transparency                 | clear behavior, visible state                                             |
| clarity                      | clear naming, clear docs                                                  |
| consistency                  | same naming, same behavior                                                |
| maintainability              | easier to change/test/read                                                |
| readability                  | easier to read                                                            |
| simplicity                   | fewer branches/files/options                                              |
| complexity                   | branches, dependencies, moving parts                                      |
| technical debt               | old code, risky code, deferred cleanup                                    |
| legacy                       | old, existing; use only if truly legacy                                   |
| migration                    | move from X to Y                                                          |
| modernization                | update from X to Y                                                        |
| seamless migration           | migration with no manual step / no breaking API                           |
| unlock value                 | make useful, expose X                                                     |
| maximize value               | improve X, increase Y                                                     |
| minimize friction            | remove steps, reduce errors                                               |
| low-hanging fruit            | easy fix                                                                  |
| quick win                    | small fix                                                                 |
| robust solution              | fix that handles X, Y, Z                                                  |
| comprehensive solution       | fix covering X, Y, Z                                                      |
| scalable solution            | design that handles N                                                     |
| elegant solution             | simple fix                                                                |
| nuanced solution             | specific fix                                                              |
| sophisticated solution       | advanced/complex fix                                                      |
| optimal solution             | best under stated metric; otherwise “good fix”                          |
| efficient solution           | faster/uses less memory                                                   |
| effective solution           | works                                                                     |
| powerful tool                | tool                                                                      |
| simple yet powerful          | simple                                                                    |
| easy-to-use                  | simple                                                                    |
| feature-rich                 | has X, Y, Z                                                               |
| fully-featured               | has X, Y, Z                                                               |
| highly configurable          | configurable                                                              |
| highly customizable          | configurable                                                              |
| robust error handling        | explicit errors, typed errors, retries                                    |
| gracefully handles errors    | catches X and returns Y                                                   |
| improve reliability          | reduce failures, add retries, add validation                              |
| improve performance          | reduce runtime, reduce memory, cache X                                    |
| improve maintainability      | split X, rename Y, remove duplication                                     |
| improve readability          | rename X, simplify Y, split Z                                             |
| improve UX                   | reduce clicks, improve labels, show errors                                |
| productionize                | add tests, config, logging, deployment                                    |
| operationalize               | run in production, schedule, monitor                                      |
| unlock potential             | delete                                                                    |
| tap into                     | use                                                                       |
| capitalize on                | use                                                                       |
| bridge the gap               | connect X and Y, fix mismatch                                             |
| meet the needs of            | support, handle                                                           |
| designed to                  | does, is                                                                  |
| aims to                      | does, should                                                              |
| seeks to                     | does, should                                                              |
| helps to                     | helps, does                                                               |
| allows users to              | lets users                                                                |
| enables developers to        | lets developers                                                           |
| provides a way to            | lets, adds                                                                |
| serves as                    | is                                                                        |
| acts as                      | is                                                                        |
| responsible for              | handles, owns                                                             |
| plays a role in              | affects, handles                                                          |
| is responsible for ensuring  | checks, enforces                                                          |
| make sure that               | make sure, check                                                          |
| in order to                  | to                                                                        |
| due to the fact that         | because                                                                   |
| with regard to               | about                                                                     |
| in the context of            | in                                                                        |
| as a result of               | because of                                                                |
| at this point in time        | now                                                                       |
| a number of                  | several, many                                                             |
| various                      | specific list, or delete                                                  |
| multiple                     | N, several                                                                |
| numerous                     | many                                                                      |
| plethora                     | many                                                                      |
| myriad                       | many                                                                      |
| utilize this approach        | use this approach                                                         |
| this approach enables        | this approach lets                                                        |
| this change improves         | this change makes X better by Y                                           |
| this implementation          | this code, this function                                                  |
| this functionality           | this feature, this behavior                                               |
| the aforementioned           | this, that                                                                |
| aforementioned               | above                                                                     |
| respectively                 | name each mapping explicitly when possible                                |
| simply                       | delete unless contrasting complexity                                      |
| clearly                      | delete unless evidence follows                                            |
| obviously                    | delete                                                                    |
| basically                    | delete                                                                    |
| actually                     | delete unless correcting something                                        |
| very                         | delete or quantify                                                        |
| significantly                | quantify or delete                                                        |
| substantially                | quantify or delete                                                        |
| drastically                  | quantify or delete                                                        |
| dramatically                 | quantify or delete                                                        |
| extremely                    | quantify or delete                                                        |
| highly                       | delete or quantify                                                        |
| easily                       | explain why or delete                                                     |
| just                         | delete unless minimizing scope intentionally                              |

Extra rule worth adding to your guide:

| Avoid                                       | Use instead                                                 |
| ------------------------------------------- | ----------------------------------------------------------- |
| vague praise                                | state the exact code change                                 |
| marketing wording                           | state the behavior                                          |
| abstract benefit                            | name the measurable effect                                  |
| generic “improvement”                     | say what changed: speed, memory, API, tests, errors, naming |
| fake certainty                              | say “likely”, “may”, or “depends on X”                |
| “best practice” without evidence          | cite the project rule, framework docs, or existing pattern  |
| “robust/scalable/optimized” without proof | give the constraint, metric, or test result                 |

### 2.2 Inflated adjectives and adverbs

Avoid: *robust, seamless, scalable, comprehensive, holistic, granular,
cutting-edge, state-of-the-art, next-generation, best-in-class, world-class,
powerful, rich, vibrant, dynamic, intricate, nuanced, profound, meaningful,
pivotal, crucial, vital, essential, paramount, significant, notable,
remarkable, invaluable, meticulous, tireless, relentless, timeless,
groundbreaking, innovative, transformative, game-changing.*

If the adjective is needed, prove it with a number or a fact:

- Bad: "A robust caching layer."
- Better: "Caching layer that survives Redis restarts and handles ~5k req/s."

If you cannot prove it, delete it.

### 2.3 Filler nouns and metaphors

Avoid: *journey, landscape, realm, world, space, ecosystem, tapestry,
symphony, kaleidoscope, fabric, lens, paradigm, frontier, era, treasure
trove, golden ticket, linchpin, cornerstone, north star.*

These words almost always mean nothing in technical text. Cut the whole
phrase, not just the word.

- Bad: "In the rapidly evolving landscape of time-series databases..."
- Better: "Time-series databases differ in three ways:" (then list them).

### 2.4 Hedging and meta-phrases

Delete on sight:

- "It is important to note that..."
- "It is worth noting that..."
- "It should be mentioned that..."
- "Please note that..."
- "Keep in mind that..."
- "As you may know..."
- "In today's fast-paced world..."
- "In an ever-evolving / ever-changing X..."
- "At its core..."
- "At the end of the day..."
- "Simply put..."
- "In essence..."
- "Ultimately..."
- "Needless to say..."

If the point matters, state it directly. If it does not matter, drop it.

### 2.5 Transition words used as filler

AI text leans on a few transitions to glue paragraphs together. Most of
them can be deleted without changing meaning.

Avoid as paragraph or sentence openers: *Furthermore, Moreover, Additionally,
Consequently, Subsequently, Nonetheless, Notwithstanding, Therefore, Thus,
Hence, Indeed, Notably.*

Use plain connectors only when the logical link is real: *and, but, so, also,
then, because, if, while.* Often the best fix is no transition at all. Start
the next sentence with the subject.

### 2.6 Marketing and corporate phrases

Never write these in this repo:

- "unlock the power of..."
- "take X to the next level"
- "game-changing solution"
- "best-in-class / world-class"
- "cutting-edge / state-of-the-art"
- "drive value / drive synergies / drive innovation"
- "mission-critical"
- "future-proof"
- "thought leadership"
- "value proposition"
- "actionable insights"
- "data-driven decisions"
- "end-to-end solution"
- "deep dive"
- "holistic approach"
- "value-added"

If the text is for a `README.md`, the rule still applies. Internal READMEs
are not sales pages.

### 2.7 First-person AI tells

Never write:

- "As an AI language model..."
- "Certainly! Here is..."
- "I hope this helps."
- "Let me know if you need anything else."
- "Great question!"
- "Of course!"
- "Sure!"

These leak into prompts, PR descriptions, and commit messages. Strip them.

---

## 3. Patterns to Avoid

### 3.1 The "not only X but also Y" pattern

- Bad: "This module not only parses the config but also validates it."
- Better: "This module parses and validates the config."

### 3.2 The rule of three (forced triplets)

AI tends to list three adjectives or three verbs even when one is enough.

- Bad: "A clean, modular, and extensible architecture."
- Better: "Each subsystem is a separate package and depends only on
  `aa_toolkit`."

### 3.3 Restating the heading

Do not open a section by paraphrasing its title.

- Bad section under `## Installation`: "This section explains how to install
  the project."
- Better: jump straight to the command.

### 3.4 Summary that adds nothing

Do not end a document with "In conclusion, ..." or "To summarize, ...". If a
summary is needed, make it a bullet list of concrete takeaways.

### 3.5 Over-explained obvious code

- Bad:
  ```python
  # Increment the counter by one
  counter += 1
  ```
- Better: delete the comment.

### 3.6 Restating the function name in the docstring

- Bad:
  ```python
  def load_config(path):
      """Load the config from the given path."""
  ```
- Better: state the format expected, the return type, and the failure mode,
  or omit the docstring if the signature already says it all.
  ```python
  def load_config(path: Path) -> Config:
      """Parse a TOML file at `path`. Raises `ConfigError` on missing keys."""
  ```

### 3.7 Speculative or aspirational text

Do not write what the code "could", "might", or "will eventually" do unless
there is a tracked task. Document what exists today.

### 3.8 Em dash and emoji habits

- Avoid em dashes (`—`) used as a stylistic break. Use a period or
  parentheses.
- No emoji in docstrings, comments, READMEs, or task files.

---

## 4. How to Write Each Artifact

### 4.1 Code comments

- Comment **why**, not **what**. The code already shows the what.
- Acceptable reasons to comment: non-obvious constraint, link to an issue or
  spec, warning about a side effect, unit of a magic number, reason a
  workaround exists.
- One line is usually enough. If a comment grows beyond ~3 lines, move it
  to a docstring or a Markdown doc.
- Do not leave commented-out code. Delete it; git keeps history.

Examples:

```python
# OPC tag uses ms since boot, not Unix time. See AA-1423.
ts = boot_ts + raw_ms / 1000

# Empirical: anything below 0.02 is sensor noise on this line.
THRESHOLD = 0.02
```

### 4.2 Docstrings

Use this shape (Google or NumPy style is fine; pick whichever the
surrounding module uses):

```python
def resample(series: pd.Series, period: str) -> pd.Series:
    """Downsample `series` to `period` using the mean of each bucket.

    NaNs are dropped before aggregation. The returned index is aligned to
    the start of each bucket.
    """
```

Rules:

- First line: one sentence, imperative mood, ends with a period.
- Skip the docstring entirely if the function name and type hints already
  say everything (private helpers usually do).
- Document parameters only when their meaning is not obvious from the name
  and type. Do not list every parameter just to fill the section.
- Document raised exceptions when the caller is expected to catch them.
- Do not repeat the type information from the signature in prose.

### 4.3 README files

Minimum useful content, in this order:

1. One sentence: what this package or app is.
2. How to install / set up (commands only, no prose around them).
3. How to run it (command + minimal example).
4. Where the entry point lives (file path + function or class).
5. Any non-obvious gotcha (env vars, ports, external services).

Skip "Features", "Why we built this", "Vision", "Roadmap" sections unless a
human asked for them.

### 4.4 `task.md` files

A `task.md` is an instruction to an agent. Write it like a ticket, not an
essay.

Required sections:

- **Goal**: one or two sentences, concrete and verifiable.
- **Scope**: files, modules, or packages allowed to change.
- **Out of scope**: what must not change.
- **Acceptance criteria**: bullet list, each item testable.
- **References**: paths to relevant code, docs, or prior tasks.

Do not write motivation or background unless the agent cannot do the task
without it. If background is needed, keep it under a `## Context` heading
and limit it to facts.

### 4.5 Agent-facing instructions (`*-RULES.md`, `AGENTS.md`)

- Use imperative voice: "Do X.", "Never write Y."
- Prefer numbered or bulleted rules over paragraphs.
- Each rule should be checkable: an agent or reviewer must be able to say
  yes or no.
- Give a short example for any rule that is easy to misread.

### 4.6 Commit messages and PR descriptions

- Subject: imperative, ≤ 72 chars, no trailing period.
  - Good: `Fix Timestream paging on empty result`
  - Bad: `Fixed an issue where the Timestream client would sometimes...`
- Body (optional): what changed and why. Skip the "what" if the diff is
  obvious.
- No greetings, no sign-offs, no thanks.

---

## 5. Bad vs. Better Examples

### 5.1 README opener

Bad:

> Welcome to the Historian Dashboard Tool! In today's fast-paced industrial
> landscape, having robust, real-time visibility into your process data is
> more crucial than ever. This powerful, comprehensive solution empowers
> engineers to unlock the full potential of their historian data through a
> seamless, intuitive interface.

Better:

> Dash app that plots tags from the Ignition historian and from AWS
> Timestream side by side. Entry point: `historian-dash-app/app.py`.

### 5.2 Module docstring

Bad:

```python
"""
This module provides a comprehensive suite of utilities to facilitate the
seamless interaction with the underlying database layer, empowering
developers to leverage powerful querying capabilities.
"""
```

Better:

```python
"""Helpers for building parameterized SQL against the historian DB."""
```

### 5.3 Function docstring

Bad:

```python
def get_user(user_id):
    """
    This function is used to get a user. It takes a user_id as input and
    returns the corresponding user object. It is a crucial part of the
    authentication flow.
    """
```

Better:

```python
def get_user(user_id: str) -> User | None:
    """Return the user with `user_id`, or `None` if not found."""
```

### 5.4 Inline comment

Bad:

```python
# Loop over all the items in the list and process each one
for item in items:
    process(item)
```

Better: delete the comment.

### 5.5 Task description

Bad:

> We need to dive deep into the existing data ingestion pipeline and
> reimagine it with a more robust, scalable, and future-proof architecture
> that empowers our team to deliver actionable insights at scale.

Better:

> **Goal:** Replace the current per-tag polling in `libs/ignitionhistorianquery.py`
> with a single batched query.
>
> **Acceptance:**
>
> - One SQL query per `fetch()` call regardless of tag count.
> - Existing unit tests pass.
> - p95 latency for 50-tag fetch under 500 ms on the dev DB.

### 5.6 Section transition

Bad:

> Furthermore, it is important to note that the cache layer plays a pivotal
> role in ensuring optimal performance.

Better:

> The cache layer holds the last 24 h of samples in memory, which removes
> ~90% of DB hits.

---

## 6. Checklist Before Submitting Any Non-Code Text

Run through this list before committing:

- [ ] No words from §2.1–2.6 unless justified by a fact or number.
- [ ] No "It is important to note", "In conclusion", "Ultimately", "At its
  core", or similar filler.
- [ ] No `Furthermore` / `Moreover` / `Additionally` as paragraph starters.
- [ ] No marketing adjectives without a measurement attached.
- [ ] No comments that restate what the next line of code does.
- [ ] No docstrings that paraphrase the function name.
- [ ] No emoji, no em-dash stylistic breaks.
- [ ] Every claim is something the reader can verify in the repo.
- [ ] The text would still make sense if read in isolation, with no
  surrounding "fluff" paragraphs.

If a sentence fails any item, rewrite it or delete it.

---

## 7. Reference: Word and Phrase Blocklist

Treat the following as a hard blocklist for this repository's documentation,
comments, docstrings, READMEs, task files, commit messages, and PR text.
They may appear inside quoted external text, dependency names, or proper
nouns, but never in our own prose.

The list is grouped by category. Some entries appear in more than one form
(verb, noun, adverb), and all forms are blocked.

### 7.1 Inflated verbs (and their `-ed` / `-ing` / `-s` forms)

action, actualize, advance, advocate for, aim, align, amplify, anchor,
architect, augment, bolster, broaden, capture, catalyze, champion, capitalize
on, cement, centralize, choreograph, circumvent, collaborate, command,
conceptualize, concretize, consolidate, construct, craft, curate, deepen,
delight, deliver, democratize, deploy, derive, devise, dial in, disrupt,
distill, dive into, double down, drive, elaborate, elevate, embark, embed,
embody, embrace, emulate, enable, enact, encapsulate, encompass, endeavor,
energize, enforce, engage, enhance, enlighten, enrich, ensure, entail,
envision, equip, espouse, evangelize, evoke, exacerbate, execute on,
exemplify, expedite, explore, extrapolate, facilitate, fast-track, finetune,
foster, frame, fuel, furnish, galvanize, gamify, gauge, generate, glean,
govern, grapple, ground, harmonize, harness, herald, highlight, hinder,
honor, ideate, illuminate, immerse, impact, implement, incentivize,
incubate, inform, ingest, innovate, inspire, instantiate, integrate,
interface, interplay, intertwine, invigorate, iterate on, kickstart,
launch, leverage, liaise, light up, lift, lock in, magnify, manifest,
materialize, maximize, mediate, meld, mentor, mobilize, model, modernize,
monetize, moonshot, motivate, navigate, nurture, observe, operationalize,
optimize, orchestrate, originate, partake, partner with, perfect, pioneer,
pivot, populate, position, power, prepare, prioritize, productize,
proliferate, propel, prosper, propagate, propose, prove out, provide,
purvey, quantify, ramp up, realize, redefine, reengineer, refine, reframe,
reimagine, rejuvenate, relate, relentlessly pursue, render, repurpose,
resolve, resonate, restate, retool, revamp, reverberate, revitalize,
revolutionize, scale, seamlessly integrate, seek, shape, shape up, shed
light, shepherd, shift, showcase, signal, simplify, socialize, solidify,
solve for, spearhead, spotlight, sprinkle, stand up, stimulate, strategize,
streamline, strengthen, stretch, strive, structure, succeed, supercharge,
support, surface, sustain, synergize, synthesize, tackle, tailor, take a
deep dive, target, thrive, tighten up, top up, touch base, transcend,
transform, transition, traverse, treat, trigger, uncover, underline,
underpin, underscore, undertake, unify, unleash, unlock, unravel,
unparalleled growth, unpack, uphold, uplift, usher in, utilize, validate,
value, venture, weave, weigh in, wield, win, work towards, wow.

### 7.2 Inflated adjectives

actionable, adept, agile, ai-driven, ai-first, ai-powered, all-encompassing,
arduous, astonishing, astounding, authentic, awe-inspiring, baked-in, basic,
battle-tested, best-of-breed, best-in-class, bespoke, blockchain-enabled,
bleeding-edge, bold, breakthrough, burgeoning, captivating, cloud-based,
cloud-first, cloud-native, cognizant, commendable, complete, complex,
compelling, comprehensive, considerable, content-rich, contextual,
contextualized, creative, critical, crucial, customer-centric, customer-first,
cutting-edge, daring, data-centric, data-driven, deep, definitive, delightful,
demonstrable, disruptive, distinguished, dynamic, effortless, elegant,
empowering, enchanting, end-to-end, enlightening, enriching, enterprise-grade,
entrenched, esteemed, ethical, ever-changing, ever-evolving, ever-expanding,
ever-growing, exceptional, exceptional quality, exceptional value, exemplary,
exciting, expansive, exquisite, extensible, extraordinary, fascinating,
finely-tuned, first-class, first-of-its-kind, flagship, flexible, flourishing,
foundational, full-fledged, full-stack, fundamental, future-forward,
future-proof, futuristic, gold-standard, granular, grand, gripping,
groundbreaking, hands-on, hardened, harmonious, head-turning, heart-pounding,
heightened, high-caliber, high-fidelity, high-impact, high-level,
high-performance, high-quality, holistic, human-centered, human-centric,
hyper-personalized, hyperscale, immersive, impactful, impressive, indelible,
industry-leading, industry-standard, ingenious, innovative, inspiring,
inspirational, integrated, intelligent, intricate, intuitive, invaluable,
iron-clad, iterative, key, laser-focused, leading, leading-edge,
lightning-fast, low-friction, low-latency, low-level, magical, magnificent,
manifold, masterful, meaningful, mesmerizing, meticulous, mind-blowing,
mission-critical, modern, modular, momentous, more than just,
multidimensional, multifaceted, must-have, native, never-before-seen,
next-gen, next-generation, next-level, niche, nimble, nontrivial, notable,
noteworthy, novel, nuanced, omnichannel, on-demand, one-stop, optimal,
out-of-the-box, outstanding, paradigm-shifting, paramount, performant,
personalized, pervasive, pioneering, pivotal, plug-and-play, polished, potent,
powerful, premier, premium, pristine, production-grade, production-ready,
profound, pronged, proven, purpose-built, rapid, rapidly evolving,
real-time, refined, relentless, remarkable, renowned, resilient, revolutionary,
rich, robust, rock-solid, scalable, seamless, secondary, self-service,
sleek, smart, sophisticated, sound, spectacular, standout, state-of-the-art,
stellar, striking, strong, stunning, substantial, superior, sustainable,
synergistic, systemic, tailored, tangible, telling, tertiary, thought-
provoking, thriving, time-tested, timeless, tireless, top-notch, top-tier,
tournament-grade, transformational, transformative, trustworthy, turnkey,
ultimate, unbeatable, unbelievable, undeniable, unforgettable, unique,
unmatched, unparalleled, unprecedented, unrivaled, untold, valuable,
value-added, various, vast, vibrant, visionary, vital, well-crafted,
well-rounded, widely recognized, world-changing, world-class.

### 7.3 Inflated adverbs

absolutely, accordingly, actively, additionally, admittedly, aptly, arguably,
broadly, broadly speaking, certainly, clearly, completely, comprehensively,
conceivably, consequently, considerably, conveniently, creatively, critically,
crucially, decidedly, deeply, definitely, demonstrably, distinctly, drastically,
dramatically, dynamically, easily, effectively, efficiently, elegantly,
emphatically, entirely, especially, essentially, evidently, exceedingly,
exceptionally, exhaustively, extensively, extraordinarily, fundamentally,
furthermore, generally, generally speaking, genuinely, gracefully,
granularly, greatly, hence, herein, heretofore, highly, holistically, however,
immensely, importantly, impressively, in earnest, increasingly, incredibly,
indeed, indubitably, inherently, intrinsically, intricately, invariably,
irrefutably, judiciously, literally, manifestly, markedly, masterfully,
materially, meaningfully, meticulously, moreover, namely, naturally,
needless to say, nevertheless, notably, notwithstanding, objectively,
obviously, painstakingly, particularly, perfectly, pivotally, plainly,
poignantly, powerfully, precisely, predominantly, preemptively, presumably,
primarily, proactively, profoundly, prominently, properly, proverbially,
quintessentially, radically, readily, really, reasonably, reliably, remarkably,
respectively, robustly, seamlessly, securely, significantly, simply, simply
put, singularly, smoothly, specifically, specifically speaking, strategically,
strikingly, subsequently, substantively, substantially, successfully,
sufficiently, surely, surprisingly, swiftly, thereby, therefore, therein,
thereof, thoroughly, thus, tirelessly, totally, truly, ubiquitously,
ultimately, unambiguously, undeniably, undoubtedly, unequivocally,
unfailingly, uniquely, universally, unmistakably, vastly, very, vibrantly,
visibly, vividly, whilst, wholly, wholeheartedly.

### 7.4 Filler nouns and metaphors

abundance, adversity, aforementioned, alchemy, arena, art, artistry,
ascent, aspiration, avenue, backbone, backdrop, balance, bandwidth, beacon,
bedrock, beginning, behemoth, bevy, blueprint, bonanza, bouquet, bounty,
bridge, brilliance, building block, bulwark, canvas, catalyst, charm,
chasm, chorus, circle, climate, cog, complexity, compass, conduit, confluence,
constellation, cornerstone, core, crescendo, crossroads, crucible, dance,
dawn, depth, destiny, digital realm, distillation, dive, dreams, drumbeat,
echo, ecosystem, edge, elixir, embodiment, endeavor, engine, enigma, epicenter,
era, essence, evolution, facet, fabric, fingerprint, fire, flagship, flair,
flagstone, flame, flavor, flicker, flourish, footprint, foothold, forefront,
foray, foundation, frontier, fulcrum, fusion, gateway, genesis, glimpse,
gold mine, golden ticket, granular detail, granular level, harbinger,
harmony, heartbeat, heart, heritage, highlight, hub, ignition, illumination,
impact, implications, inception, ingredient, insight, insights, inspiration,
intersection, journey, juncture, kaleidoscope, keystone, labyrinth, landscape,
leap, legacy, lens, lifeblood, lifeline, lifecycle, lifeforce, life, light,
linchpin, magic, magnitude, manifold, masterpiece, melody, melting pot,
microcosm, milestone, mission, modus operandi, moment, momentum, mosaic,
muse, narrative, nexus, north star, nucleus, nuance, oasis, odyssey,
overture, paradigm, paradigm shift, passion, pathway, patchwork, peak,
philosophy, pillar, pinnacle, pioneer, pivot, plethora, plot twist, polish,
portrait, possibilities, potential, powerhouse, precipice, precursor,
prelude, prism, promise, quest, radiance, realm, recipe, rebirth, regimen,
reimagining, renaissance, rendezvous, renewal, reservoir, revelation,
revival, revolution, ripple, ripple effect, rising star, rite of passage,
roadmap, scheme, sea change, secret sauce, sentinel, shape, signal, silver
bullet, smorgasbord, snapshot, song, soul, source, space, spark, spectrum,
spirit, springboard, stage, sterling reputation, story, stride, stronghold,
sweep, symphony, syntax, tableau, tangle, tapestry, terrain, testament,
the future of, the linchpin of, the next frontier, the power of, the road
ahead, threshold, tide, titan, toolkit, torchbearer, touchpoint, transformation,
trove, treasure trove, triumph, trove, twist, undercurrent, underpinning,
universe, uncharted territory, uncharted waters, utmost, vanguard, veil,
verge, vessel, vibe, vision, voyage, wave, wealth, web, wellspring, whisper,
whole, wisdom, world, zenith, zest.

### 7.5 Marketing and corporate jargon

24/7 support, ai-driven solution, ai-first, ai-native, ai-powered, agile
methodology, alignment, all-hands, at scale, audience engagement, back-office,
bandwidth (as in capacity), best practices, best-in-class, big picture,
biz dev, blue-sky thinking, boil the ocean, boots on the ground, bottom line,
brand awareness, brand equity, brand voice, break down silos, bring to the
table, business agility, business continuity, business outcomes, capability,
capacity building, change agent, change management, circle back, cloud-first,
cloud journey, cloud-native solution, collaborative environment, competitive
advantage, competitive landscape, compliance posture, content strategy,
continuous improvement, core competency, corporate social responsibility,
cost optimization, cost-effective, customer 360, customer journey, customer
loyalty, customer satisfaction, customer-centric, customer-first, dashboard
of dashboards, data-driven, data-driven decision, decision-makers, deep
dive, deep understanding, delight customers, deliverables, deliver value,
deployment plan, devops culture, digital fluency, digital journey, digital
transformation, disrupt the market, disruptive innovation, domain expertise,
double-click on, drill down, drive synergies, drive value, driven approach,
driving innovation, due diligence, dynamic environment, eat our own dog food,
ecosystem play, efficiency gains, elevator pitch, embrace change, emerging
technologies, employee engagement, end-to-end solution, enterprise-grade,
enterprise readiness, ethical considerations, expertise, fail fast, faster
time-to-market, first-mover advantage, foster innovation, fresh perspectives,
from inception to execution, frictionless, full-stack solution, future-proof,
game changer, game-changer, get our ducks in a row, go-to-market, golden
record, governance framework, green-field, growing recognition, growth hack,
growth mindset, hand-in-glove, healthy pipeline, high-performing team,
high-touch, hit the ground running, holistic approach, human capital,
implementation strategy, in the weeds, industry best practices, industry-
leading, influencers, innovation pipeline, intelligent automation, internal
stakeholders, intuitive ux, issue resolution, journey mapping, key learnings,
key takeaways, knowledge transfer, kpis, leading indicator, lessons learned,
level set, leverage synergies, lift and shift, line of sight, low-hanging
fruit, market fit, market penetration, market share, market trends,
mission-aligned, mission-critical, mission statement, modernization journey,
moonshot, move the needle, mvp, net new, new heights, new normal,
next-generation, north star metric, offboarding, offerings, omnichannel
experience, on the same page, onboarding, open the kimono, operational
efficiency, operational excellence, opportunity space, organic growth,
out of pocket, paddle in the same direction, pain point, paradigm shift,
peel the onion, performance optimization, ping me, pivot strategy,
poc, power user, problem solving, process optimization, productize,
profitability, push the envelope, put a pin in it, quality assurance,
quality control, quick win, raise the bar, rapidly evolving market,
reaching new heights, regulatory compliance, resource allocation, resource
optimization, return on investment, revenue growth, risk mitigation, roadmap,
roi, root cause analysis, rubber meets the road, run it up the flagpole,
sandbox, scale at speed, scrum, secret sauce, seamless experience,
self-service, service excellence, single pane of glass, single source of
truth, sla, smart solution, solution development, solutioning, source of
truth, sprint, stakeholders, strategic alignment, strategic imperative,
strategic initiative, strong presence, subject matter experts, success
criteria, sustainability, swim lane, synergies, synergistic, synergistically,
synergy, table stakes, take it offline, tco, team player, technology stack,
the new normal, the next frontier, thought leaders, thought leadership,
thought partner, time optimization, time-to-market, time-to-value, top of
mind, total cost of ownership, touch base, transforming the way, trickle
down, trusted advisor, turnkey solution, uplift, uptime, user adoption,
user engagement, user experience, user feedback, user interface, user
journey, value add, value chain, value creation, value driver, value
proposition, value-added, vendor agnostic, viral growth, vision and
strategy, voice of the customer, walk the talk, white-glove, white-space
opportunity, win-win, world-class team.

### 7.6 Hedging, meta-commentary, and filler openers

a host of, a journey of, a multitude of, a myriad of, a plethora of, a
testament to, a wealth of, above all, after all, all in all, all things
considered, ample opportunities, an array of, as a general rule, as a
matter of fact, as a result, as a side note, as discussed, as far as X is
concerned, as mentioned above, as mentioned earlier, as mentioned previously,
as previously stated, as such, as we all know, as we have seen, as we move
forward, as you can see, as you may already know, at first glance, at its
core, at its heart, at length, at scale, at the end of the day, at the
heart of, at the same time, based on the information provided, be that as
it may, before delving into, beyond a shadow of a doubt, broadly speaking,
by all means, by and large, by extension, by the same token, by way of
example, can be seen as, cannot be overstated, certainly, certainly here
are, certainly here is, certainly here's, despite the fact that, due to the
fact that, demonstrates significant, encountered hurdles, ever-changing
landscape, ever-evolving landscape, fast-paced world, first and foremost,
for all intents and purposes, for example, for instance, for the most part,
for what it's worth, generally speaking, given that, going forward, granted
that, here are some, here is a, here we go, here's the deal, here's the
kicker, here's the thing, hopefully this helps, however, i hope this clears
things up, i hope this helps, i would like to, if i may, if you will, in
a manner of speaking, in a nutshell, in a sea of, in addition, in addition
to that, in any case, in brief, in conclusion, in contrast, in detail, in
effect, in essence, in fact, in general, in great detail, in light of, in
my humble opinion, in my opinion, in my view, in no small part, in no small
way, in order to, in other words, in particular, in passing, in practice,
in reality, in regard to, in retrospect, in short, in some sense, in some
ways, in sum, in summary, in summation, in terms of, in the dynamic world
of, in the end, in the final analysis, in the fast-paced world of, in the
following section, in the grand scheme of things, in the long run, in the
realm of, in theory, in this article, in this context, in this day and age,
in this section, in today's day and age, in today's fast-paced world, in
today's rapidly evolving market, in today's world, indeed, insights into,
interestingly enough, it bears mentioning, it cannot be denied, it cannot
be overstated, it goes without saying, it has been observed that, it is
clear that, it is essential, it is evident that, it is important to consider,
it is important to keep in mind, it is important to mention, it is important
to note, it is interesting to note, it is no surprise that, it is often
said, it is widely accepted that, it is widely known that, it is worth
mentioning, it is worth noting, it should be emphasized, it should be
mentioned, it should be noted, it stands to reason, it's clear that, it's
crucial to, it's important to, it's important to consider, it's important to
note, it's important to remember, it's no secret that, it's safe to say,
it's well known that, it's worth mentioning, it's worth noting, just to be
clear, last but not least, let me explain, let's break it down, let's dive
in, let's dive into, let's explore, let's get into it, let's take a look,
look no further, looking ahead, looking forward, more often than not,
moving forward, navigating the complexities of, navigating the landscape,
needless to say, no doubt, not to mention, notably, now, more than ever,
of course, offer a comprehensive, on a final note, on a side note, on the
ascent to, on the contrary, on the cutting edge, on the flip side, on the
one hand, on the other hand, on top of that, one might argue, overall,
particularly in areas, please note, plus, prior to, put simply, quite
frankly, rest assured, shedding light on, should you have any questions,
showcasing, significantly contributes, similarly, simply put, since the
dawn of, so without further ado, sometimes, that being said, that said,
the fact of the matter is, the long and short of it, the truth is, this
article will explore, this paper presents, this section will, time and time
again, to be clear, to be honest, to be sure, to begin with, to clarify,
to conclude, to demonstrate, to elaborate, to elevate, to elucidate, to
emphasize, to empower, to enhance, to enrich, to exemplify, to facilitate,
to furnish, to give an example, to highlight, to illustrate, to make a
long story short, to maximize, to my knowledge, to provide, to put it
another way, to put it bluntly, to put it simply, to recap, to reiterate,
to say the least, to shed light on, to showcase, to start with, to state
the obvious, to sum up, to summarize, to thrive, to top it off, to
underscore, to unleash, to unlock, today's, truly, ultimately, understanding
of your unique, very, well, well-crafted, what's more, when all is said and
done, when it comes to, when push comes to shove, while it is true, whilst
it is true, with a keen eye on, with regard to, with regards to, with that
being said, with that in mind, without a doubt, without further ado,
without question, you might be wondering, you see.

### 7.7 AI first-person tells (never appear in our text)

absolutely, certainly i can help, certainly here is, dive into the topic of,
ever-evolving world of, glad to help, great question, here is a comprehensive,
here you go, hope this helps, i am an ai, i am happy to, i am here to assist,
i can definitely help, i hope this answers your question, i hope this clears
things up, i hope this email finds you well, i hope this gives you a good
overview, i hope this helps, i must clarify, i understand that, i would be
happy to, i would love to, i'd be happy to, i'm an ai language model, i'm
glad you asked, i'm here to help, i'm just a language model, i'm not able
to, i'm sorry but, i'm sorry for any confusion, i'm sorry for the
confusion, i'm sorry i cannot, in this article we will explore, it would be
my pleasure, let me clarify, let me know if i can help further, let me
know if there is anything else, let me know if you have any other questions,
let me know if you need anything else, let me know if you need more, let
me walk you through, let's break this down, let's dive deeper into, let's
dive in, let's dive into, let's embark on this journey, let's explore,
let's explore this topic, let's get started, my apologies for, of course,
of course i can, please feel free to, please let me know, please note that,
sure, sure thing, sure here is, thank you for asking, thank you for the
question, thanks for your patience, that's a great question, that is a
great question, that's an excellent question, the topic of, this is a
fascinating topic, today we will explore, welcome to my, without further
ado.

### 7.8 Words that often need a fact attached (use only with a number or

citation)

These are not banned outright, but never use them as bare claims. If you
write one of these, the same sentence must include a measurement, version,
file path, benchmark, or external reference that justifies it.

accuracy, adoption rate, aligns, availability, backward compatibility,
better, bigger, capacity, compatibility, complexity, conducting, consistency,
correctness, coverage, deployment plan, downtime, efficiency, expertise,
faster, fewer, granular, growth, idle, improvement, iteration, latency,
maintainability, maximize, memory, more, optimize, overhead, performance,
precision, productivity, quality, recall, reduce, reliability, resilience,
response time, scalable, security, simpler, slower, smaller, speed,
stability, stronger, throughput, uptime, utilization.

Bad: "The new layer is more scalable."
Good: "The new layer handles 5× the previous request rate (see
`tests/load/test_layer.py`)."

Bad: "Faster query path."
Good: "Query path drops p95 from 820 ms to 140 ms on the dev DB
(`benchmarks/query.py`)."

### 7.9 Punctuation, formatting, and typographic tells

Avoid these in our prose:

- Em dash (`—`) used as a stylistic break. Use a period or parentheses.
- Curly quotes (`“”` `‘’`). Use straight quotes.
- Ellipsis as suspense (`...`). State the next sentence directly.
- Title Case Headings For Every Word. Use sentence case.
- Lists of exactly three items where one would do.
- Bold or italic for emphasis on filler words ("**very** important").
- Emoji in docstrings, comments, READMEs, task files, or commit messages.
- Trailing exclamation marks.
- Rhetorical questions ("Why does this matter?", "Sound familiar?",
  "Ever wondered…?").
- Markdown horizontal rules (`---`) used for decoration between unrelated
  short paragraphs.

### 7.10 Sentence shapes to avoid

- "Not only X, but also Y." → "X and Y."
- "X is more than just Y." → state what X is.
- "It's not about X, it's about Y." → say Y.
- "X is the new Y." → delete.
- "Imagine a world where..." → delete.
- "What if I told you..." → delete.
- "From X to Y, ..." (as opener) → start with the subject.
- "When it comes to X, ..." → start with the subject.
- "At the heart of X lies Y." → "X uses Y." or "X depends on Y."
- "X plays a crucial role in Y." → say what X actually does in Y.
- "X serves as a Y." → "X is a Y." (if true) or describe behavior.
- "This is where X comes in." → introduce X directly.
- "Picture this:" → delete.

### 7.11 Quick-reference rule

When in doubt, delete the word and reread the sentence. If the sentence
still conveys the same fact, the word was filler. If the sentence collapses,
rewrite it with a concrete noun, verb, number, or file reference instead of
the blocked term.
