# Architecture Rules

## Repository Architecture

This repository is organized as a curriculum, not as an application stack.

- Topic folders are the primary architectural units.
- Individual notebooks are the primary implementation units.
- `Figures/` stores reusable visual assets for notebooks.
- `Literature/` stores the canonical source material that notebook content must align with.
- Root documentation files define how agents should operate and how notebooks should be authored.

### Directory Responsibilities

- [01_Foundations](/home/almuno/github/optimal-control-ml-notebooks/01_Foundations): prerequisite mathematics and supporting theory.
- [02_Dynamics](/home/almuno/github/optimal-control-ml-notebooks/02_Dynamics): modeling of dynamical systems.
- [03_Optimization](/home/almuno/github/optimal-control-ml-notebooks/03_Optimization): optimization formulations and methods.
- [04_Optimal_Control](/home/almuno/github/optimal-control-ml-notebooks/04_Optimal_Control): control design and optimal control methods.
- [05_Machine_Learning](/home/almuno/github/optimal-control-ml-notebooks/05_Machine_Learning): ML topics that should remain compatible with the earlier math and control material.
- [07_Reinforcement_Learning](/home/almuno/github/optimal-control-ml-notebooks/07_Reinforcement_Learning): RL topics that connect back to dynamics, optimization, and control.

### Dependency Boundaries

- Do not introduce service layers, managers, or helper frameworks for notebook content unless explicitly requested.
- Keep lesson-specific code in the notebook that teaches it.
- Reuse figures and literature references instead of duplicating assets or inventing parallel terminology.
- Use the existing scientific Python stack unless the task explicitly requires something else.
- Preserve relative linking between neighboring notebooks; notebook navigation is part of the architecture.

---

## **8. MANDATORY NOTEBOOK ARCHITECTURE (NON-NEGOTIABLE)**

All notebooks in this repository **MUST** follow the **exact structure** defined below.  
This architecture ensures pedagogical consistency, readability, and progression from beginner to advanced control engineers.

**PRINCIPLES:**
- **Single, Strict Structure**: Every notebook follows the same section order, regardless of topic.
- **Separation of Concerns**: Theory, assumptions, model definition, implementation, experiments, results, and conclusions are clearly separated.
- **LLM-Generatable**: The structure is explicit enough that an LLM can generate conforming notebooks without human correction.
- **Scalability**: Works for simple foundational topics (vector operations) and complex control topics (MPC, Kalman filtering, trajectory optimization).

Repository-specific interpretation:

- Existing notebooks under foundations already demonstrate this pattern and should be the reference style for future work.
- Optional cells are optional only when they genuinely add no pedagogical value.
- If a notebook uses figures, those figures should be stored under `Figures/` and linked relatively from the notebook.

---

### **8.1. Universal Notebook Structure**

Every notebook **MUST** contain the following sections **in this exact order**:

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

### **8.2. Cell-by-Cell Specification**

#### **CELL 1: Title and Navigation Context**
**Type:** Markdown  
**Purpose:** Establish the topic's identity, numbering, and hierarchical position.

**REQUIRED ELEMENTS:**
1. Section number following the repository hierarchy (e.g., `1.1.2.3`, `4.7.2.1`)
2. Topic title (concise, descriptive)
3. NO introduction text, NO preamble

**Format Template:**
```markdown
### <section_number>. <Topic Title>
```

**Examples:**
```markdown
### 1.1.2.3. Dot Product
```
```markdown
### 4.7.1.2. Linear Quadratic Regulator (LQR)
```

**FORBIDDEN:**
- Introductory sentences ("In this notebook, we will...")
- Learning objectives lists
- Prerequisites sections
- Table of contents

---

#### **CELL 2: Mathematical Definition**
**Type:** Markdown  
**Purpose:** Present the core mathematical formulation(s) central to the topic.

**REQUIRED ELEMENTS:**
1. Primary equation(s) in LaTeX (display mode `$$...$$`)
2. NO explanatory text in this cell (explanation comes later)
3. Variable definitions **only if essential** for equation clarity

**Format Template:**
```markdown
$$
<primary_equation>
$$
```

**Examples:**

**Simple (Foundations):**
```markdown
$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i
$$
```

**Complex (Control):**
```markdown
$$
\begin{aligned}
\min_{\mathbf{u}_0, \ldots, \mathbf{u}_{N-1}} \quad & \mathbf{x}_N^T \mathbf{P} \mathbf{x}_N + \sum_{k=0}^{N-1} \left( \mathbf{x}_k^T \mathbf{Q} \mathbf{x}_k + \mathbf{u}_k^T \mathbf{R} \mathbf{u}_k \right) \\
\text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{A} \mathbf{x}_k + \mathbf{B} \mathbf{u}_k, \quad k = 0, \ldots, N-1 \\
& \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max} \\
& \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max}
\end{aligned}
$$
```

**FORBIDDEN:**
- Inline explanations
- Multiple unrelated equations
- Code snippets

---

#### **CELL 3: Theoretical Explanation**
**Type:** Markdown  
**Purpose:** Provide conceptual understanding, intuition, and practical context.

**REQUIRED ELEMENTS:**
1. **"Explanation:"** heading
2. Concise conceptual description (2-4 sentences)
3. **[OPTIONAL]** "Assumptions:" subsection (for complex topics)
4. **[OPTIONAL]** "Properties:" subsection (for mathematical objects)
5. **"Example:"** heading with analytical or numerical demonstration

**Format Template:**
```markdown
**Explanation:**

<2-4 sentences explaining the concept, its purpose, and relevance to ML/control>

[OPTIONAL for complex topics:]
**Assumptions:**
- Assumption 1
- Assumption 2

[OPTIONAL for mathematical objects:]
**Properties:**
- Property 1
- Property 2

**Example:**

<Analytical or numerical example with step-by-step derivation>

If
$$
<setup>
$$

then
$$
<result>
$$
```

**Examples:**

**Simple (Vector Norm):**
```markdown
**Explanation:**

The Euclidean norm measures the magnitude (length) of a vector.  
It is useful in optimization and distance computations in ML.

**Example:**

If  
$$
\mathbf{v} =
\begin{bmatrix}
3 \\
4
\end{bmatrix},
$$  

then  
$$
\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5.
$$
```

**Complex (Kalman Filter):**
```markdown
**Explanation:**

The Kalman filter is an optimal state estimator for linear dynamical systems with Gaussian noise.  
It recursively computes the minimum mean-square error estimate of the state given noisy measurements.  
Widely used in navigation, robotics, and signal processing.

**Assumptions:**
- Linear system dynamics: $\mathbf{x}_{k+1} = \mathbf{A} \mathbf{x}_k + \mathbf{B} \mathbf{u}_k + \mathbf{w}_k$
- Linear measurement model: $\mathbf{y}_k = \mathbf{C} \mathbf{x}_k + \mathbf{v}_k$
- Process noise $\mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q})$
- Measurement noise $\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R})$
- Initial state estimate known

**Example:**

Consider a 1D constant-velocity model tracking a moving object.  
State: $\mathbf{x} = [position, velocity]^T$  
Measurement: position only, corrupted by noise with $\sigma = 5$ m.
```

**FORBIDDEN:**
- Implementation details
- Code snippets
- Lengthy derivations (keep concise)
- Duplicate equations from Cell 2

---

#### **CELL 4: Implementation**
**Type:** Python Code  
**Purpose:** Provide minimal, elegant, self-explanatory implementation.

**REQUIRED ELEMENTS:**
1. Necessary imports (grouped at top)
2. Parameter/data definition
3. Core computation
4. Output display (print or return)

**STYLE RULES:**
- Follow all coding rules in [CODING-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/CODING-RULES.md)
- NO inline comments explaining "what" (code must be self-explanatory)
- NO docstrings
- Prefer comprehensions over loops (Rule 7)
- Use clearly named variables (Rule 2)

**Format Template:**
```python
import <library>

# Define inputs
<input_definition>

# Compute result
<computation>

# Display output
print("<label>", result)
```

**Examples:**

**Simple (Dot Product):**
```python
import numpy as np

u = np.array([1, 2])
v = np.array([3, 4])

result = np.dot(u, v)

print("u · v =", result)
```

**Complex (LQR):**
```python
import numpy as np
from scipy.linalg import solve_discrete_are, inv

A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.5], [1.0]])
Q = np.eye(2)
R = np.array([[0.1]])

P = solve_discrete_are(A, B, Q, R)
K = inv(R + B.T @ P @ B) @ B.T @ P @ A

print("Optimal gain K =\n", np.round(K, 4))
```

**FORBIDDEN:**
- Comments explaining obvious operations
- Defensive checks (unless explicitly requested)
- Multiple unrelated experiments in one cell
- Placeholder values (use realistic parameters)

---

#### **CELL 5: [OPTIONAL] Advanced Analysis/Visualization**
**Type:** Python Code  
**Purpose:** Provide visual insight, comparative analysis, or extended experiments.

**WHEN TO INCLUDE:**
- Visualization enhances understanding (e.g., phase portraits, convergence plots)
- Comparing multiple methods/parameters
- Demonstrating robustness or sensitivity
- Complex topics requiring intuition-building (MPC, trajectory optimization)

**WHEN TO OMIT:**
- Simple algebraic operations (vector addition, matrix transpose)
- Visualization adds no pedagogical value

**Format Template:**
```python
import matplotlib.pyplot as plt

# Experiment setup
<parameters>

# Run simulation/comparison
<computation_loop>

# Plot results
plt.figure(figsize=(8, 5))
<plotting_commands>
plt.xlabel("<label>")
plt.ylabel("<label>")
plt.title("<title>")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Examples:**

**Visualization (Matrix Inverse Transformation):**
```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([1.0, 1.0])
A = np.array([[2, 1], [0, 1]])
A_inv = np.linalg.inv(A)

v_transformed = A @ v
v_back = A_inv @ v_transformed

plt.figure(figsize=(8, 5))
plt.plot([0, v[0]], [0, v[1]], label="Original v", linewidth=2.5)
plt.plot([0, v_transformed[0]], [0, v_transformed[1]], label="Transformed Av", linewidth=2.5)
plt.plot([0, v_back[0]], [0, v_back[1]], label="After inverse A⁻¹(Av)", linewidth=2, linestyle="--")
plt.scatter([v[0], v_transformed[0], v_back[0]], [v[1], v_transformed[1], v_back[1]], s=80)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Matrix Inverse Representation")
plt.show()
```

**Comparative Analysis (MPC Horizon Study):**
```python
import numpy as np
import matplotlib.pyplot as plt
from control_library import simulate_mpc

horizons = [5, 10, 20, 50]
results = {N: simulate_mpc(A, B, Q, R, N, x0) for N in horizons}

plt.figure(figsize=(10, 6))
for N, trajectory in results.items():
    plt.plot(trajectory[:, 0], label=f"N={N}")
plt.xlabel("Time step")
plt.ylabel("State x₁")
plt.title("MPC Performance vs Prediction Horizon")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**FORBIDDEN:**
- Redundant plots (multiple views of same data)
- Unformatted axes (always label and title)
- Code unrelated to the topic

---

#### **CELL 6: [OPTIONAL] Extended Theory or Variants**
**Type:** Markdown  
**Purpose:** Present alternative formulations, extensions, or related concepts.

**WHEN TO INCLUDE:**
- Multiple equivalent formulations exist (e.g., batch vs recursive least squares)
- Important variants (e.g., discrete vs continuous LQR)
- Extensions (e.g., constrained vs unconstrained optimization)

**WHEN TO OMIT:**
- No meaningful variants exist
- Topic is introductory and extensions would confuse

**Format Template:**
```markdown
**[Variant/Extension Name]:**

<Brief explanation>

$$
<alternative_formulation>
$$

<1-2 sentences on when to use this variant>
```

**Example:**

**Variant (Recursive Least Squares):**
```markdown
**Recursive Formulation:**

Instead of batch processing, the least squares solution can be updated sequentially as new data arrives:

$$
\begin{aligned}
\mathbf{K}_k &= \mathbf{P}_{k-1} \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_{k-1} \mathbf{H}_k^T + \mathbf{R})^{-1} \\
\hat{\mathbf{x}}_k &= \hat{\mathbf{x}}_{k-1} + \mathbf{K}_k (\mathbf{y}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k-1}) \\
\mathbf{P}_k &= (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k-1}
\end{aligned}
$$

Useful for online estimation and memory-constrained systems.
```

---

#### **CELL 7: References and Navigation**
**Type:** Markdown  
**Purpose:** Credit sources and enable repository navigation.

**REQUIRED ELEMENTS:**
1. **"References:"** heading
2. Linked reference(s) to source material (books, papers, courses)
3. Horizontal rule `---`
4. Navigation links: `[⬅️ Previous: <title>](<path>) | [Next: <Next Topic> ➡️](<path>)`

**Format Template:**
```markdown
**References:**

[📘 <Author>. (<Year>). *<Title>*](<URL>)
[📗 <Author>. (<Year>). *<Title>*](<URL>)

---

[⬅️ Previous: <Previous Topic>](<relative_path>) | [Next: <Next Topic> ➡️](<relative_path>)
```

**Examples:**

**Single Reference:**
```markdown
**References:**

[📘 Aazi, M. (2024). *Mathematics For Machine Learning*](https://www.scribd.com/document/812294393/Mathematics-for-Machine-Learning)

---

[⬅️ Previous: Dot Product](./03_vector_dot_product.ipynb) | [Next: Norm ➡️](./05_vector_norm.ipynb)
```

**Multiple References:**
```markdown
**References:**

[📘 Aazi, M. (2024). *Mathematics For Machine Learning*](https://www.scribd.com/document/812294393/Mathematics-for-Machine-Learning)  
[📗 Rozycki, P. (2020). *Computational Mechanics Course Notes, École Centrale de Nantes*](https://www.ec-nantes.fr/study/masters/computational-mechanics)

---

[⬅️ Previous: Determinant](./09_matrix_determinant.ipynb) | [Next: Rank–Nullity ➡️](./11_matrix_rank_nullity.ipynb)
```

**EMOJIS:**
- Every reference line MUST start with a book emoji: `📘` for primary textbooks, `📗` for supplementary sources (lecture notes, course material, other references).
- The "Previous" navigation link MUST start with `⬅️`: `[⬅️ Previous: <Title>](<path>)`
- The "Next" navigation link MUST end with `➡️`: `[Next: <Title> ➡️](<path>)`
- These emojis are part of the mandatory format and must not be omitted.

**LINKS:**
- Reference links MUST point to a publicly accessible online URL (publisher page, DOI, PDF, official course page, etc.).
- References MUST NOT link to local paths inside the `Literature/` folder. The `Literature/` directory is a local working resource and is not pushed to the repository; local links will be broken for every reader.

**FORBIDDEN:**
- Unreferenced content (all notebooks must cite sources)
- Broken navigation links
- Missing previous/next links (except for first/last in section)
- Reference lines without a book emoji (`📘` or `📗`)
- Navigation links without directional emojis (`⬅️`, `➡️`)
- Reference links pointing to local `Literature/` paths (use a public URL instead)

---

### **8.3. What Changes Per Notebook vs What Stays Identical**

#### **ALWAYS IDENTICAL (Structure):**
1. Number of mandatory cells (Cells 1-4, 7)
2. Cell order
3. Section headings ("Explanation:", "Example:", "References:")
4. Navigation link format

#### **CHANGES PER TOPIC (Content):**
1. **Section number** (reflects hierarchy: `01_Foundations/01_Linear_Algebra/02_Vector/03_vector_dot_product.ipynb` → `1.1.2.3`)
2. **Topic title** (e.g., "Dot Product" vs "Kalman Filter")
3. **Equations** (specific to the concept)
4. **Explanation text** (tailored to topic complexity)
5. **Implementation code** (different algorithms, parameters, data)
6. **Visualization** (if Cell 5 exists, customized to topic)
7. **References** (cite relevant sources)
8. **Navigation paths** (previous/next notebook paths)

---

### **8.4. Special Rules for Different Topic Areas**

#### **Foundations (01_Foundations/):**
- Keep examples **simple and numerical** (small matrices, 2D/3D vectors)
- Emphasize **computational verification** (show algebraic and coded results match)
- **Minimal** or **no** Cell 5 (visualization) unless it aids intuition (e.g., geometric interpretations)

#### **Dynamics (02_Dynamics/):**
- **Cell 3** MUST include "Assumptions:" (e.g., small angles, linearization)
- **Cell 4** MUST define system parameters explicitly (mass, stiffness, damping)
- **Cell 5** SHOULD include time-domain simulation plots

#### **Optimization (03_Optimization/):**
- **Cell 2** MUST show objective function and constraints
- **Cell 3** MUST explain feasibility and optimality conditions
- **Cell 5** SHOULD compare solution methods or visualize feasible regions

#### **Optimal Control (04_Optimal_Control/):**
- **Cell 3** MUST state control objective (regulation, tracking, etc.)
- **Cell 4** MUST define dynamics, cost matrices, horizons
- **Cell 5** MUST show closed-loop response plots
- **Cell 6** SHOULD discuss stability, robustness, or extensions

#### **Machine Learning / Reinforcement Learning (05_ML, 07_RL):**
- **Cell 3** MUST explain model assumptions (i.i.d. data, stationarity, etc.)
- **Cell 4** MUST include dataset generation or loading
- **Cell 5** MUST show training curves, performance metrics, or decision boundaries

---

### **8.5. LLM Generation Checklist**

When generating a new notebook, verify:

- [ ] Section number matches folder hierarchy
- [ ] Cell 1 contains only `### <number>. <Title>`
- [ ] Cell 2 contains only equations (no prose)
- [ ] Cell 3 has "Explanation:" and "Example:" headings
- [ ] Cell 4 follows coding rules (Sections 0-7)
- [ ] Cell 5 exists only if visualization/analysis adds pedagogical value
- [ ] Cell 6 exists only if meaningful variants/extensions exist
- [ ] Cell 7 has references and navigation links
- [ ] Code is self-explanatory (no comments explaining "what")
- [ ] All variables have descriptive names (no `x`, `y`, `tmp`)
- [ ] Comprehensions used where appropriate (Rule 7)
- [ ] No defensive checks unless explicitly required (Rule 0)

---

### **8.6. Enforcement**

**This structure is MANDATORY.**  
Any notebook violating this architecture will be flagged for correction.  
LLMs generating notebooks MUST produce conforming output on first attempt.

**When in doubt:**
- Prefer **simplicity** over complexity (KISS)
- Prefer **conciseness** over verbosity
- Prefer **explicit structure** over clever formatting

---

## **9. LITERATURE FOLDER AS CANONICAL REFERENCE (NON-NEGOTIABLE)**

The `Literature/` directory is the **source of truth** for all notebook content.

Current repository note:

- The literature tree contains both PDFs and Markdown conversions.
- Some chapters already have `.md` files and metadata, while others currently exist only as PDFs or placeholders.
- Agents should use the best available source in `Literature/`, audit Markdown conversions for extraction errors, and avoid inventing terminology when a canonical source already exists.

### **9.1. Structure and Purpose**

The `Literature/` folder contains Markdown files converted from books via PDF→Markdown pipeline, organized by book and chapter:

```text
Literature/
├── 01 - Foundations/
│   ├── Aazi 2024 - Mathematics For Machine Learning/
│   │   ├── Chapter01_Linear_Algebra/
│   │   ├── Chapter02_Probability_and_Statistics/
│   │   └── ...
│   ├── Savov 2016 - No bullshit guide to linear algebra/
│   │   ├── Savov 2016 - No bullshit guide to linear algebra - chapter 01/
│   │   └── ...
├── 02 - Dynamical Systems/
├── 03 - Optimization/
├── 04 - Control/
└── ...
```

### **9.2. Mandatory Compliance Rules**

1. **Vocabulary Consistency:**  
   All notebooks MUST use vocabulary, definitions, and explanations that **match** the corresponding literature files.

2. **Structure Mirroring:**  
   Notebooks MUST directly reference, mirror structure, and stay consistent with the corresponding literature chapters.

3. **Terminology as Source of Truth:**  
   The literature files define canonical terminology and structure-notebooks MUST NOT diverge from them.

4. **Conversion Error Auditing:**  
   Because PDF→Markdown conversion may introduce errors, any extracted text or concept MUST be **audited for correctness and consistency** before being used in notebooks.

### **9.3. Workflow**

When creating or updating a notebook:

1. **Locate** the corresponding literature file (e.g., `Literature/01 - Foundations/Savov 2016 - .../chapter_02.md`)
2. **Extract** relevant definitions, explanations, and examples
3. **Audit** extracted content for conversion errors (OCR mistakes, malformed equations, missing symbols)
4. **Adapt** content to notebook structure (Cells 1-7) while preserving original vocabulary
5. **Reference** the source in Cell 7

### **9.4. Forbidden**

- Using definitions or terminology that contradict the literature files
- Inventing explanations when literature provides canonical text
- Ignoring available literature and writing from scratch
- Using unaudited extracted content with potential conversion errors

---

Formatting is **strict** and must always follow this structure:

---
### ✅ Output Format

1. **Markdown Theory Section**  
    _(Ensure clear paragraph spacing for readability)_
    - Main equation (if applicable)
    - Concise, literature-accurate or summarized explanation (keep same terminology)
    - Analytical example (if applicable)
    - Numerical example (if applicable)
        
2. **Code Section**
    - Short, elegant implementation
    - No extra comments or blank lines beyond what’s necessary
        
3. **References + Links**
    - Include book, author, or chapter reference when available
        
4. **Updated Previous Slide Link**
    - Always include the link or reference to the previous notebook section
        

---

Each new notebook derived from a screenshot must strictly adhere to this format.

Example:

```Markdown
### 1.1.1.3 Dot Product


$$

\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n

$$


**Explanation:**


The dot product calculates a scalar representing the magnitude of projection of one vector onto another.

It is widely used in machine learning for similarity measures or linear operations.

  
**Example:**

  
If


$$

\mathbf{u} =

\begin{bmatrix}

1 \\

2

\end{bmatrix},

\quad

\mathbf{v} =

\begin{bmatrix}

3 \\

4

\end{bmatrix},

$$

  
then

  
$$

\mathbf{u} \cdot \mathbf{v} = 1 \cdot 3 + 2 \cdot 4 = 11.

$$
```

Implementation

```python
import numpy as np

# Define vectors
u = np.array([1, 2])
v = np.array([3, 4])

# Compute dot product
result = np.dot(u, v)
print("u · v =", result)
```

```Markdown
**References:**

[📘 Aazi, M. (2024). *Mathematics For Machine Learning*](https://www.scribd.com/document/812294393/Mathematics-for-Machine-Learning)

---

[⬅️ Previous: Scalar Multiplication](./02_vector_scalar_multiplication.ipynb) | [Next: comming soon ➡️]()
```


Previous slide update:
```Markdown
**References:**

[📘 Aazi, M. (2024). *Mathematics For Machine Learning*](https://www.scribd.com/document/812294393/Mathematics-for-Machine-Learning)

  
---
  

[⬅️ Previous: Vector Addition](./01_vector_addition.ipynb) | [Next: Dot Product ➡️](./03_vector_dot_product.ipynb)
```

---

## **10. CONCEPT CROSS-REFERENCING AND NOTEBOOK WRITING STYLE (NON-NEGOTIABLE)**

Notebooks should include **natural references to related concepts** that exist elsewhere in the repository. Explanations must be **author-neutral, concept-centered, and pedagogically direct**.

### **10.1. Purpose**

- Improve conceptual navigation across the curriculum.
- Help readers move between prerequisite and advanced concepts.
- Maintain a coherent conceptual graph across the repository.

### **10.2. Cross-Referencing Guidelines**

- References must appear **naturally within the explanation**.
- Do **not force references** just to create links.
- Only link concepts that are **meaningfully used in the explanation**.
- Avoid listing references without context.
- Prefer links embedded in explanatory text.
- Do not overlink foundational concepts that are obvious or unrelated to the immediate explanation.

### **10.3. Link Format**

Use standard relative notebook links:

```markdown
[linear transformation](./01_linear_transformation.ipynb)
```

### **10.4. Do Not Reference Authors in Explanations**

Notebook explanations must **not directly mention authors or books** when explaining concepts. The notebooks teach the concept itself, not how a particular author introduces it.

**Bad:**

```markdown
Fleisch begins Chapter 4 with the change in vector components caused by rotating the coordinate axes.
```

```markdown
Savov defines this concept as the transformation of coordinates between reference frames.
```

Reason: these statements add no conceptual value and introduce unnecessary author commentary.

**Good:**

```markdown
When the coordinate axes rotate, the numerical components of a vector change even though the geometric vector itself remains the same.
```

### **10.5. Do Not Describe the Literature Structure**

Notebooks must **not reference chapters, sections, or how books organize topics**.

**Bad:**

```markdown
Savov places rotations and reflections under this broader concept.
```

```markdown
Fleisch discusses this topic in Chapter 4.
```

Reason: this is metadata about the book, not part of the mathematical explanation.

### **10.6. Do Not Justify Links Using Author Commentary**

References to other notebooks must **not be introduced through statements about how an author organized the topic**.

**Bad:**

```markdown
Savov places rotations and reflections under this broader concept, so this notebook links to the matrix representations of [rotations](../06_Linear_Transformations/11_rotation_matrix_representation.ipynb) and [reflections](../06_Linear_Transformations/10_reflection_matrix_representation.ipynb) rather than re-explaining them geometrically.
```

Reason: the link is justified by discussing the author's organization instead of the concept itself.

**Good:**

```markdown
Rotations and reflections are specific examples of linear transformations. Their matrix representations are explored in the notebooks on [rotation matrices](../06_Linear_Transformations/11_rotation_matrix_representation.ipynb) and [reflection matrices](../06_Linear_Transformations/10_reflection_matrix_representation.ipynb).
```

### **10.7. Use Neutral Concept-Driven Explanations**

All explanations must focus on:

- The mathematical idea.
- The geometric intuition.
- The algebraic representation.
- The relationship between concepts.

Avoid meta-discussion about sources.

### **10.8. References to Literature Are Internal Only**

The literature in `Literature/` should guide **content accuracy**, but it must not appear in the explanatory text.

Authors and books may only appear in:

- Repository documentation.
- Bibliography sections (Cell 7).
- Reference metadata (if explicitly required).

They must **not appear inside concept explanations**.

### **10.9. Examples of Good Cross-References (Natural Integration)**

**Example 1:**

```markdown
**Explanation:**

Composing two linear transformations means applying one and then the other. The abstract composition law becomes ordinary matrix multiplication once both transformations are written in matrix form. This depends on the definition of a [linear transformation](./01_linear_transformation.ipynb) and the concept of [matrix representation](./05_matrix_representation.ipynb).
```

**Example 2:**

```markdown
**Explanation:**

A change of basis rewrites vectors relative to a new coordinate system. This process is closely tied to the concept of a [basis](./02_basis_definition.ipynb) and to the matrix form of [coordinate transformations](./03_change_of_basis_matrix.ipynb).
```

**Example 3:**

```markdown
**Explanation:**

Matrix-vector multiplication can be interpreted geometrically as applying a linear transformation to a vector. This interpretation follows directly from the definition of a [linear transformation](../06_Linear_Transformations/01_linear_transformation.ipynb).
```

### **10.10. Examples of Bad References (Forced, Artificial, or Author-Driven)**

**Example 1:**

```markdown
Builds on [Vectors](./01_vector_definition.ipynb). Together with [Scalar Multiplication of a Vector](./03_vector_scalar_multiplication.ipynb), this operation underlies the linear combinations used in [Matrix-Vector Multiplication](../03_Matrix/04_matrix_vector_multiplication.ipynb).
```

Reason: references are listed mechanically rather than appearing naturally within the explanation.

**Example 2:**

```markdown
This topic uses concepts from [Vectors](./01_vector_definition.ipynb), [Matrices](../03_Matrix/01_matrix_definition.ipynb), and [Linear Transformations](../06_Linear_Transformations/01_linear_transformation.ipynb).
```

Reason: generic reference dumping without explanatory integration.

**Example 3:**

```markdown
Before studying this concept, review [Vector Addition](./02_vector_addition.ipynb), [Scalar Multiplication](./03_vector_scalar_multiplication.ipynb), and [Linear Combination](./04_linear_combination.ipynb).
```

Reason: prerequisite list inserted artificially instead of being referenced where conceptually needed.

**Example 4:**

```markdown
Savov places rotations and reflections under this broader concept, so this notebook links to [rotations](../06_Linear_Transformations/11_rotation_matrix_representation.ipynb) and [reflections](../06_Linear_Transformations/10_reflection_matrix_representation.ipynb).
```

Reason: link justified by author commentary instead of being introduced through the concept.

### **10.11. Final Rule**

Notebook explanations must be **author-neutral, concept-centered, and pedagogically direct**. Do not mention authors, chapters, or book structure. Concept references should **enhance understanding**, appear **naturally within the explanation**, and **never be inserted mechanically just to create links** or justified by commentary about the literature.
