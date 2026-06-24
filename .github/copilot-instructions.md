Below is your content rewritten with **proper Markdown indentation, spacing, and formatting**, while preserving **all original meaning and rules exactly**.

---

# Strict Coding Rules

> **Principle:** *Always prefer simplicity. Clear, simple logic beats clever, complex solutions.*

---

## 0. MOST IMPORTANT RULE (NON-NEGOTIABLE):

## DO NOT OVERENGINEER — DO NOT ADD UNREQUESTED FEATURES / CHECKS / EXCEPTIONS

Respect these principles **NO MATTER WHAT**:

* **SOC** — Separation of Concerns
* **DYC** — Default to the path of least complexity
* **DRY** — Don’t Repeat Yourself
* **KISS** — Keep It Simple, Stupid
* **YAGNI** — You Aren’t Gonna Need It

If a change conflicts with any of these, **do not make it**.

---

### Core Rules

* **Implement exactly what the prompt asks — nothing more.**
  No “future-proofing” or “safety nets” unless explicitly required.

* **No speculative validations or branching.**
  Do **not** add guard clauses, boolean branches, or defensive checks (e.g., `if A or B or C or D`) unless explicitly requested.

* **Keep concerns separated (SOC).**
  Do not move logic across modules or introduce new architectural layers.

* **Avoid duplication (DRY)** only when it does not introduce abstraction beyond the prompt.
  Deduplicate locally and plainly.

* **Prefer the simplest working code (KISS).**
  The smallest change that fulfills the prompt is the correct change.

* **Do not build for hypothetical futures (YAGNI).**
  If it’s not explicitly requested now, don’t implement it now.

* **DYC applies at decision points.**
  Choose the path that reduces conditionals and complexity while meeting the prompt exactly.

---

### What “Explicitly Requested” Means

* The prompt **states the check or feature by name or behavior**.
* Vague fears such as “what if X is None?” are **not** permission to add code.

---

### Consequences of Violating Rule 0

The following will be removed if unrequested:

* Added checks
* Flags
* Retries
* Confirmations
* Caching
* Type gates
* Undo logic
* Permission scaffolding
* Feature toggles
* Service objects
* Managers
* Config plumbing

---

### Examples

#### ❌ Bad (Unrequested Branching)

```python
# Prompt: Add a delete button for notes in the sidebar.
def delete_note(note_id):
    if note_id is None or note_id == "" or note_id not in notes or not isinstance(note_id, str):
        return
    if user_is_admin() and has_permission("delete"):
        notes.remove(note_id)
```

#### ✅ Good (Exact Scope, Minimal)

```python
def delete_note(note_id):
    notes.remove(note_id)
```

---

#### ❌ Bad (Unrequested Multi-Condition Safety Net)

```python
# Prompt: Inline rename field.
def rename(note_id, new_title):
    if new_title is None or new_title.strip() == "" or len(new_title) > 256 or contains_html(new_title):
        return
    if not isinstance(note_id, str):
        return
    notes[note_id].title = new_title
```

#### ✅ Good (Prompt-Only Behavior)

```python
def rename(note_id, new_title):
    notes[note_id].title = new_title
```

---

### Mini-Checklist (Must Pass Before Submitting)

* Did I implement **only** what the prompt asked?
* Did I avoid adding new branches, guards, flags, configs, or services?
* Did I keep logic within the existing module’s responsibility (SOC)?
* Is the change the **simplest possible** (KISS)?
* Did I avoid “future” features (YAGNI)?
* Are all original behaviors identical unless explicitly changed?

---

# 1. Code Style and Structure

## Code Must Be Self-Explanatory

---

## Avoid Code Duplication (DRY)

#### ❌ Bad

```python
if condition1:
    process_data(data)
elif condition2:
    process_data(data)
```

#### ✅ Good

```python
def handle_data(condition, data):
    if condition:
        process_data(data)
```

---

## Avoid Deep Nesting (Arrow Code)

#### ❌ Bad

```python
for item in items:
    if item.valid:
        for sub_item in item:
            if sub_item.active:
                process(sub_item)
```

#### ✅ Good

```python
active_sub_items = [
    sub_item
    for item in items if item.valid
    for sub_item in item.sub_items if sub_item.active
]

for active_sub_item in active_sub_items:
    process(active_sub_item)
```

---

# 2. Naming Conventions

* **Single-character names are prohibited.**
* Variables must be descriptive and intent-revealing.

---

### Explicit Naming

#### ❌ Bad

```python
for a in b:
    c(a)
```

#### ✅ Good

```python
for user in user_list:
    send_email(user)
```

---

### Avoid Abbreviations

#### ❌ Bad

```python
def calc(r, h):
    return 3.14 * r ** 2 * h
```

#### ✅ Good

```python
def calculate_cylinder_volume(radius, height):
    return math.pi * radius ** 2 * height
```

---

# 3. Comments and Docstrings (High Priority)

* Code must be self-explanatory.
* **Do NOT add docstrings.**
* **Do NOT add comments explaining “what.”**
* Comments are allowed only if explaining **why**, and only when necessary.

---

#### ❌ Bad

```python
# add user to list
users.append(user)
```

#### ✅ Good

```python
# required because external API expects non-empty user list
users.append(user)
```

---

# 4. Error Handling and Conditionals

## Avoid Excessive try-except

#### ❌ Bad

```python
try:
    val = int(user_input)
except:
    val = 0
```

#### ✅ Good

```python
if user_input.isdigit():
    val = int(user_input)
else:
    handle_invalid_input()
```

---

## Avoid Unnecessary Fallbacks

#### ❌ Bad

```python
def get_value(key):
    return data.get(key, "")
```

#### ✅ Good

```python
def get_value(key):
    if key in data:
        return data[key]
    handle_missing_key(key)
```

---

## Avoid Introspection

#### ❌ Bad

```python
if isinstance(obj, list):
    process_list(obj)
```

#### ✅ Good

```python
process_iterable(obj)
```

---

# 5. Readability and Single Responsibility

Each function must have one clear purpose.

#### ❌ Bad

```python
def process_user(user):
    verify_user(user)
    send_email(user)
    update_database(user)
```

#### ✅ Good

```python
def process_user(user):
    verify_user(user)
    notify_user(user)
    save_user(user)
```

---

# 6. Never Use Complex Operations in Loop Headers

### ❌ Bad Examples

```python
for idx, (x, y) in enumerate(zip(x_list, y_list)):
for user in sorted(set(users)):
for item in data[::2]:
for sum in (a+b for a, b in zip(x,y)):
for row in (line.strip().split(',') for line in file):
```

---

### ✅ Good Practice

```python
sorted_users = sorted(set(users))

for user in sorted_users:
    notify(user)
```

---

# 7. List Comprehensions (MOST IMPORTANT)

Always prefer comprehensions unless a for-loop is clearly faster.

---

#### ❌ Bad

```python
result = [s for i in items if i.valid for s in i.sub_items if s.active]
```

#### ✅ Good

```python
active_sub_items = [
    sub_item
    for item in items if item.valid
    for sub_item in item.sub_items if sub_item.active
]
```

Always format vertically for clarity.

---

# 8. SymPy as Primary Implementation Library (Non-Negotiable)

**Always prefer SymPy** for notebook implementations. SymPy produces exact symbolic results that match the mathematical definitions in Cell 2, making the output look like textbook mathematics.

Use NumPy or SciPy **only** when:
- The task is purely numerical (large-scale simulation, optimization solvers, data processing).
- Performance matters and symbolic computation is impractical.
- A specific numerical routine has no symbolic equivalent.

For everything else — derivations, definitions, algebraic manipulation, equation display — **SymPy is the default**.

### Symbol Naming Rules

Symbols **must** use single mathematical letters or standard notation so that printed output reads like literature.

#### ❌ Bad (programming-style names)

```python
import sympy as sp

vec_x = sp.Matrix([sp.Symbol('vec_x1'), sp.Symbol('vec_x2')])
mat_A = sp.Matrix([[sp.Symbol('a11'), sp.Symbol('a12')],
                    [sp.Symbol('a21'), sp.Symbol('a22')]])
result = mat_A * vec_x
```

#### ✅ Good (mathematical symbols)

```python
import sympy as sp

x_1, x_2 = sp.symbols('x_1 x_2')
a, b, c, d = sp.symbols('a b c d')

x = sp.Matrix([x_1, x_2])
A = sp.Matrix([[a, b],
               [c, d]])

result = A * x
result
```

#### ❌ Bad (verbose names for standard quantities)

```python
eigenvalue_1, eigenvalue_2 = sp.symbols('eigenvalue_1 eigenvalue_2')
state_vector = sp.Matrix([sp.Symbol('state_1'), sp.Symbol('state_2')])
```

#### ✅ Good (standard mathematical notation)

```python
lambda_1, lambda_2 = sp.symbols('lambda_1 lambda_2')
x = sp.Matrix([sp.symbols('x_1 x_2')])
```

#### ✅ Good (Greek letters and subscripts)

```python
alpha, beta, gamma = sp.symbols('alpha beta gamma')
theta, phi, psi = sp.symbols('theta phi psi')
omega_n, zeta = sp.symbols('omega_n zeta')
sigma_1, sigma_2 = sp.symbols('sigma_1 sigma_2')
```

### Common Naming Conventions

| Concept | Symbol | SymPy |
|---|---|---|
| Scalar variables | $x, y, z$ | `x, y, z = sp.symbols('x y z')` |
| Matrix entries | $a_{ij}$ | `a, b, c, d = sp.symbols('a b c d')` |
| Eigenvalues | $\lambda_i$ | `lambda_1, lambda_2 = sp.symbols('lambda_1 lambda_2')` |
| Time | $t$ | `t = sp.symbols('t')` |
| Frequency | $\omega$ | `omega = sp.symbols('omega')` |
| State vector | $\mathbf{x}$ | `x = sp.Matrix([x_1, x_2])` |
| Control input | $\mathbf{u}$ | `u = sp.Matrix([u_1, u_2])` |
| Cost/objective | $J$ | `J = sp.Symbol('J')` |
| Damping ratio | $\zeta$ | `zeta = sp.Symbol('zeta')` |
| Natural frequency | $\omega_n$ | `omega_n = sp.Symbol('omega_n')` |

### Output Display

Prefer `result` (last expression in cell) or `sp.pprint(result)` over `print(result)` so Jupyter renders the output as formatted mathematics.

---

# 9. Mandatory Notebook Architecture (Non-Negotiable)

Every notebook must follow this exact structure:

```markdown
# CELL 1: Title and Navigation Context (Markdown)
# CELL 2: Mathematical Definition (Markdown)
# CELL 3: Theoretical Explanation (Markdown)
# CELL 4: Implementation (Python Code)
# CELL 5: [OPTIONAL] Advanced Analysis/Visualization (Python Code)
# CELL 6: [OPTIONAL] Extended Theory or Variants (Markdown)
# CELL 7: References and Navigation (Markdown)
```

---

## 9.1. Cell Requirements Summary

### CELL 1 — Title

```markdown
### <section_number>. <Topic Title>
```

No intro text.

---

### CELL 2 — Mathematical Definition

Only primary equation(s):

```markdown
$$
<primary_equation>
$$
```

No prose.

---

### CELL 3 — Theoretical Explanation

Must include:

* **Explanation:**
* **Example:**
* Optional **Assumptions**
* Optional **Properties**

No code.

---

### CELL 4 — Implementation

Must include:

* Imports
* Inputs
* Computation
* Output

No docstrings.
No unnecessary comments.
No defensive checks.

---

### CELL 5 — Optional Visualization

Only when pedagogically useful.
Must include labeled axes and title.

---

### CELL 6 — Optional Extensions

Alternative formulations or variants only.

---

### CELL 7 — References and Navigation

Must include:

* **References**
* Horizontal rule `---`
* Navigation links

---

# 10. Literature Folder as Canonical Reference (Non-Negotiable)

The `Literature/` directory is the **source of truth**.

### Mandatory Compliance

1. Vocabulary must match literature.
2. Structure must mirror literature.
3. Terminology must not diverge.
4. Extracted text must be audited for conversion errors.

---

### Workflow

1. Locate literature file.
2. Extract definitions.
3. Audit for errors.
4. Adapt to notebook structure.
5. Reference source in Cell 7.

---

### Forbidden

* Inventing explanations when literature exists.
* Ignoring available literature.
* Using unaudited extracted content.

---

# Final Enforcement Rule

This structure is mandatory.

When in doubt:

* Prefer simplicity (KISS)
* Prefer conciseness
* Prefer explicit structure
* Avoid cleverness
* Implement only what is requested

---

If you'd like, I can now convert this into a **clean README.md template**, a **PDF-ready version**, or a **lint-enforceable style guide version**.
