# Repository Context

## Purpose

This repository is a notebook-first learning hub for mathematics, dynamics, optimization, optimal control, machine learning, and reinforcement learning.

The primary output of the repository is not a library or application. It is a structured curriculum of Jupyter notebooks that explain concepts, show the governing mathematics, and include small executable implementations or visualizations.

## Project Goals

- Build a coherent learning path from linear algebra foundations to advanced control and learning topics.
- Keep every notebook pedagogically consistent so a reader can move through the curriculum without adapting to new formats.
- Ground notebook content in canonical source material stored under `Literature/`.
- Use minimal code to support understanding, not to introduce software complexity.
- Preserve a clean mapping between folder hierarchy, section numbering, topic naming, references, and notebook navigation.

## Current State Of The Repository

The repository already contains substantial notebook coverage under [01_Foundations](/home/almuno/github/optimal-control-ml-notebooks/01_Foundations), especially linear algebra. The other top-level tracks already exist as directories and should be treated as part of the intended curriculum:

- [01_Foundations](/home/almuno/github/optimal-control-ml-notebooks/01_Foundations)
- [02_Dynamics](/home/almuno/github/optimal-control-ml-notebooks/02_Dynamics)
- [03_Optimization](/home/almuno/github/optimal-control-ml-notebooks/03_Optimization)
- [04_Optimal_Control](/home/almuno/github/optimal-control-ml-notebooks/04_Optimal_Control)
- [05_Machine_Learning](/home/almuno/github/optimal-control-ml-notebooks/05_Machine_Learning)
- [07_Reinforcement_Learning](/home/almuno/github/optimal-control-ml-notebooks/07_Reinforcement_Learning)

Supporting material lives in:

- [Figures](/home/almuno/github/optimal-control-ml-notebooks/Figures): reusable images referenced from notebooks.
- [Literature](/home/almuno/github/optimal-control-ml-notebooks/Literature): PDFs and Markdown conversions that define the canonical terminology and source material.
- [requirements.txt](/home/almuno/github/optimal-control-ml-notebooks/requirements.txt): the baseline notebook toolchain, currently centered on `numpy`, `sympy`, `scipy`, `matplotlib`, `pandas`, and Jupyter packages.

## Conceptual Model

Each notebook is a single lesson in a larger sequence.

- The folder path defines the domain and subdomain.
- The filename order defines the local lesson sequence.
- The notebook title cell exposes the canonical section number and topic title.
- The references/navigation cell connects the lesson to its literature source and to adjacent lessons.

Agents should think of the repository as a structured textbook with executable examples, not as a general-purpose software project.

## Domain Model

The intended curriculum progresses in this order:

1. Mathematical foundations that support later topics.
2. Dynamical systems modeling.
3. Optimization methods.
4. Optimal control methods.
5. Machine learning and reinforcement learning methods that build on the earlier math and control foundations.

This means explanations, notation, and examples in later folders should remain compatible with the earlier notebooks instead of redefining concepts arbitrarily.

## Notebook-First Mentality

When making changes, optimize for:

- pedagogical clarity
- consistency with neighboring notebooks
- literature-aligned terminology
- minimal code that demonstrates the idea directly

Do not optimize for:

- reusable software frameworks
- abstract service layers
- speculative utilities
- general infrastructure unless the prompt explicitly asks for it

## Sources Of Truth

Use the following order of trust:

1. Direct user request.
2. [AGENTS.md](/home/almuno/github/optimal-control-ml-notebooks/AGENTS.md)
3. [CODING-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/CODING-RULES.md)
4. [ARCH-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/ARCH-RULES.md)
5. Existing nearby notebooks, figures, and references.
6. Canonical literature under [Literature](/home/almuno/github/optimal-control-ml-notebooks/Literature).

If a notebook change touches mathematical definitions, explanation wording, notation, or examples, consult the matching literature source before editing.

## Repository-Specific Expectations For Agents

- Read neighboring notebooks before adding or changing a lesson.
- Preserve numbering consistency between folder hierarchy and notebook title.
- Preserve previous/next navigation consistency.
- Reuse existing figures when appropriate instead of creating redundant variants.
- Keep code examples small enough to fit the instructional purpose of a single notebook cell.
- Treat Markdown-converted literature as useful but auditable; conversion artifacts are possible.
