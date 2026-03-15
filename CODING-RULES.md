Strictly follow the coding rules outlined below.

> **Principle:** _Always prefer simplicity. Clear, simple logic beats clever, complex solutions._

These rules apply to notebook code cells, supporting scripts, and documentation examples in this repository. Because this repository is notebook-first, code should teach the concept directly and avoid introducing software architecture that does not serve the lesson.

---
## **0. MOST IMPORTANT RULE (NON-NEGOTIABLE): DO NOT OVERENGINEER — DO NOT ADD UNREQUESTED FEATURES/CHECKS/EXCEPTIONS**

Respect these principles **NO MATTER WHAT**: **SOC**, **DYC**, **DRY**, **KISS**, **YAGNI**.  
If a change conflicts with any of these, **do not make it**.

- **Implement exactly what the prompt asks—nothing more.**  
    No “future-proofing,” or “safety nets” unless the prompt explicitly requires them.
    
- **No speculative validations or branching.**  
    Do **not** add extra guard clauses, boolean branches, or “defensive” checks (e.g., `if A or B or C or D ...`) unless the prompt **explicitly** requests those checks. Assume inputs and preconditions as described; do not expand the scope.
    
- **Keep concerns separated (SOC).**  
    Do not move logic across modules or introduce new layers. Use existing module responsibilities exactly as defined.
    
- **Avoid duplication (DRY) only when it doesn’t create abstractions beyond the prompt.**  
    Deduplicate locally and plainly—do not introduce frameworks, managers, or patterns.
    
- **Prefer the simplest working code (KISS).**  
    The smallest change that fulfills the prompt is the correct change.
    
- **Do not build for hypothetical futures (YAGNI).**  
    If it’s not explicitly requested now, don’t implement it now.
    
- **DYC applies at decision points.**  
    When in doubt, choose the path that **reduces** conditionals, options, and complexity while still meeting the prompt exactly.
    

### What “explicitly requested” means

- The prompt **states the check/feature by name or behavior**.
    
- Vague fears (“what if X is None?”) are **not** permission to add code.
    

### Consequences of violating Rule 0

- Any added checks, flags, retries, confirmations, caching, type gates, undo logic, permission scaffolding, feature toggles, service objects, “managers,” or config plumbing **will be removed**.
    

### Examples

**Bad (unrequested branching):**

```python
# Prompt: Add a delete button for notes in the sidebar.
def delete_note(note_id):
    if note_id is None or note_id == "" or note_id not in notes or not isinstance(note_id, str):
        return  # ❌ unrequested guard rails
    if user_is_admin() and has_permission("delete"):  # ❌ scope creep
        notes.remove(note_id)
```

**Good (exact scope, minimal):**

```python
def delete_note(note_id):
    notes.remove(note_id)
```

**Bad (unrequested multi-condition safety net):**

```python
# Prompt: Inline rename field.
def rename(note_id, new_title):
    if new_title is None or new_title.strip() == "" or len(new_title) > 256 or contains_html(new_title):
        return                       # ❌ speculative validations
    if not isinstance(note_id, str):
        return                       # ❌ speculative type checks
    notes[note_id].title = new_title
```

**Good (prompt-only behavior):**

```python
def rename(note_id, new_title):
    notes[note_id].title = new_title
```

### Mini-Checklist (must pass before submitting)

-  Did I implement **only** what the prompt asked?
-  Did I avoid adding new branches/guards/flags/configs/services?
-  Did I keep logic within the existing module’s responsibility (SOC)?
-  Is the change the **simplest** possible (KISS) and free of duplication bloat (DRY)?
-  Did I avoid features “for later” (YAGNI)?
-  Are all original behaviors and outputs **identical** unless the prompt explicitly changed them?

---

## **1. Code Style and Structure**

- **CODE MUST BE SELF-EXPLANATORY**
- **Avoid Code Duplication (DRY: Don’t Repeat Yourself)**  
    Bad practice:
    ```python
    # Repetitive logic
    if condition1:
        process_data(data)
    elif condition2:
        process_data(data)
    ```
    
    Good practice:
    ```python
    # Clear abstraction
    def handle_data(condition, data):
        if condition:
            process_data(data)
    ```
    
- **Avoid Deep Nesting ("Arrow code")**  
    Bad practice (Arrow code):
    ```python
    for item in items:
        if item.valid:
            for sub_item in item:
                if sub_item.active:
                    process(sub_item)
    ```
    
    Good practice:
```python
	active_sub_items = [
	    sub_item 
	    for item in items if item.valid 
	    for sub_item in item.sub_items if sub_item.active
	]
	
	for active_sub_item in active_sub_items:
	    process(active_sub_item)
```

- `item` clearly refers to each element in `items`.
- `sub_item` explicitly denotes elements within each `item` (e.g., `item.sub_items`).
- `active_sub_items` precisely describes the resulting list, making the logic fully self-explanatory.

Additional repository guidance:

- Prefer short notebook implementations that demonstrate the math directly.
- Use imports already present in [requirements.txt](/home/almuno/github/optimal-control-ml-notebooks/requirements.txt) unless the prompt explicitly requires a new dependency.
- Keep one conceptual computation per notebook code cell unless the notebook architecture explicitly calls for a comparison or visualization.

---

##  **2. Naming Conventions**

- **Single-character or abbreviated variable names (e.g., `x`, `y`, `tmp`) are strictly prohibited. Variables must be words.**
    
- **Explicit Naming (Intent-Revealing Names)**  
    Bad practice:
    ```python
    for a in b:
        c(a)
    ```
    
    Good practice:
    ```python
    for user in user_list:
        send_email(user)
    ```
    
- **Avoid Abbreviations or Ambiguities**  
    Bad practice:
    ```python
    def calc(r, h):
        return 3.14 * r ** 2 * h
    ```
    
    Good practice:
    ```python
    def calculate_cylinder_volume(radius, height):
        return math.pi * radius ** 2 * height
    ```
    
- In notebook code, variable names should reflect the math without becoming cryptic.
- For matrices and vectors that conventionally use symbols like `A`, `B`, `Q`, `R`, or `K`, those standard symbols are acceptable when the topic is mathematically defined that way.

---

##  **3. Comments and Docstrings** (HIGH PRIORITY)

- **CODE MUST BE SELF-EXPLANATORY**
- **DO NOT ADD DOC-STRINGS** 
- **DO NOT ADD COMMENTS** 
- **IF COMMENTS ARE ADDED: No Comments Explaining "What"; Only if they Explain "Why" (Minimalistic Commenting)** AND ONLY IF THE IMPLEMENTATION IS NOT OBVIOUS  
    Bad practice:
    ```python
    # add user to list
    users.append(user)
    ```
    
    Good practice:
    ```python
    # required here because external API expects non-empty user list
    users.append(user)  
    ```
    
- In notebook cells, prefer markdown for explanation and keep the code cell itself clean.

---

##  **4. Error Handling and Conditionals**

- **CODE MUST BE SELF-EXPLANATORY**
- **Avoid Excessive Use of `try-except` ("Pokemon Exception Handling")**  
    Bad practice:
    ```python
    try:
        val = int(user_input)
    except:
        val = 0
    ```
    
    Good practice:
    ```python
    if user_input.isdigit():
        val = int(user_input)
    else:
        handle_invalid_input()
    ```
    
- **Avoid Unnecessary Fallbacks (Explicit Handling)**  
    Bad practice:
    ```python
    def get_value(key):
        return data.get(key, "")
    ```
    
    Good practice:
    ```python
    def get_value(key):
        if key in data:
            return data[key]
        handle_missing_key(key)
    ```
    
- **Avoid Introspection ("Duck Typing over Introspection")**  
    Bad practice:
    ```python
    if isinstance(obj, list):
        process_list(obj)
    ```
    
    Good practice:
    ```python
    process_iterable(obj)  # relies on duck typing
    ```
    
- In this repository, most notebook examples should assume the stated mathematical inputs and avoid defensive scaffolding unless the prompt specifically requests robustness analysis or error handling.

---

## **5. Readability and Clarity (Single Responsibility Principle)**

- **CODE MUST BE SELF-EXPLANATORY**    
- Each function or code block must have a clear, singular purpose.

Bad practice:
```python
def process_user(user):
    verify_user(user)
    send_email(user)
    update_database(user)
```

Good practice:
```python
def process_user(user):
    verify_user(user)
    notify_user(user)
    save_user(user)
```

- In notebooks, this usually means one code cell should compute one main result, or one visualization cell should show one well-defined analysis.

---

##  **6. NEVER Use Complex Operations in Loop Headers**

Never execute complex operations directly in loop declarations. Separate these clearly beforehand.

**Examples of Bad Practices:**

- **Complex Tuple Unpacking**
    ```python
    for idx, (x, y) in enumerate(zip(x_list, y_list)):
    ```
    
- **Nested Function Calls or Chained Methods**
    ```python
    for user in sorted(set(users)):
    ```
    
- **Inline Indexing or Slicing**
    ```python
    for item in data[::2]:
    ```
    
- **Inline Math or Logic Operations**
    ```python
    for sum in (a+b for a, b in zip(x,y)):
    ```
    
- **Multiple Inline Transformations**
    ```python
    for row in (line.strip().split(',') for line in file):
    ```
    

**Recommended Good Practice:**  
Use clearly named intermediate variables:
```python
# clearly named intermediate variable
sorted_users = sorted(set(users))
for user in sorted_users:
    notify(user)
```

---
## **7. List Comprehensions MOST IMPORTANT RULE DO NOT IGNORE!!!

IMPORTANT, Always Prefer Comprehensions over for loops, unless a for loop would be faster 
 
- **Use comprehensions, we prefer them, but use them when they clearly simplify the logic and enhance readability.**
- **Structure comprehensions explicitly and vertically for clarity.**

**Bad Practice (Inline, complex comprehension):**
```python
result = [s for i in items if i.valid for s in i.sub_items if s.active]
```

**Good Practice (Explicit, vertical structure):**
```python
active_sub_items = [
    sub_item
    for item in items if item.valid
    for sub_item in item.sub_items if sub_item.active
]
```

**Explanation:**
- Clearly named intermediate variables (`item`, `sub_item`) are preferred.
- Vertical layout helps readability, especially with multiple conditions or loops.
Always prioritize readability and explicitness in comprehensions—when in doubt, use vertical formatting.

**REMEMBER:**  
**GOOD CODE IS SELF-EXPLANATORY!**
