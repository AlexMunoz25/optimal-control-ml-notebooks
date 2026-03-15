# Agent Workflow

This file is the entry point for agents working in this repository.

Before implementing any change, read these files in order:

1. [CONTEXT.md](/home/almuno/github/optimal-control-ml-notebooks/CONTEXT.md)
2. [CODING-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/CODING-RULES.md)
3. [ARCH-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/ARCH-RULES.md)


## How Agents Must Operate

- Assume this is a notebook-first curriculum repository, not a general-purpose software service.
- Prefer minimal changes that preserve the existing pedagogical structure.
- Keep terminology, notation, and explanations aligned with nearby notebooks and canonical literature.
- Do not introduce incompatible architecture, helper layers, or speculative abstractions.
- Preserve references, numbering, figure usage, and previous/next navigation whenever a notebook is changed.

## Required Workflow Before Editing

1. Read [CONTEXT.md](/home/almuno/github/optimal-control-ml-notebooks/CONTEXT.md), [CODING-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/CODING-RULES.md), and [ARCH-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/ARCH-RULES.md).
2. Inspect the target file and the nearest related files.
3. If the task touches notebook content, inspect adjacent notebooks in the same folder to preserve numbering, navigation, notation, and difficulty level.
4. If the task touches theory, definitions, or terminology, locate the corresponding source under [Literature](/home/almuno/github/optimal-control-ml-notebooks/Literature) and audit it before reuse.
5. Implement the smallest change that satisfies the request.
6. Verify the result against the coding rules and the notebook architecture.
7. Confirm that links, references, section numbers, and figures still resolve correctly.

## Notebook-Specific Checklist

Use this checklist whenever adding or editing a notebook:

- Confirm the folder hierarchy matches the section number used in Cell 1.
- Keep the mandatory cell order defined in [ARCH-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/ARCH-RULES.md).
- Keep markdown explanation in markdown cells and implementation in code cells.
- Keep code self-explanatory, minimal, and free of unrequested guard rails.
- Cite the canonical source material in the references cell.
- Update previous/next navigation for the edited notebook and any directly affected neighbor notebooks.

## Architecture Guardrails

- Do not move notebook content into external modules unless the prompt explicitly asks for shared code.
- Do not create managers, services, wrappers, or configuration plumbing for simple notebook tasks.
- Do not invent terminology when canonical literature already provides it.
- Do not add optional notebook sections unless they materially improve pedagogy for that topic.

## When Rules Conflict

Use this precedence:

1. Direct user request
2. [AGENTS.md](/home/almuno/github/optimal-control-ml-notebooks/AGENTS.md)
3. [CODING-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/CODING-RULES.md)
4. [ARCH-RULES.md](/home/almuno/github/optimal-control-ml-notebooks/ARCH-RULES.md)
5. [CONTEXT.md](/home/almuno/github/optimal-control-ml-notebooks/CONTEXT.md)

If a requested change would break the repository architecture or notebook workflow, call that out explicitly before making the change.
