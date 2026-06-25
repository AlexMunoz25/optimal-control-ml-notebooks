# 03_Calculus — Track Architecture

Calculus foundation track for `01_Foundations`. This file is the canonical plan for the track: it fixes the subfolders, the notebook sequence, the section numbering, the literature mapping, and the computational stack so future agents can author conforming notebooks without re-deriving the design.

> **Status:** architecture only. No notebooks have been created yet. Build them in the order below.

Authoring rules are **not** restated here. Every notebook must follow the root guidance exactly:
[AGENTS.md](../../AGENTS.md) · [CONTEXT.md](../../CONTEXT.md) · [CODING-RULES.md](../../CODING-RULES.md) · [ARCH-RULES.md](../../ARCH-RULES.md) (mandatory 8-cell structure, §8.4 *Foundations* rules apply).

---

## Purpose

Provide exactly the differential and integral calculus that the repository's later domains depend on — process modeling, optimization, optimal control, machine learning, and reinforcement learning — and nothing beyond those learning goals. The track culminates in the gradient/Jacobian/Hessian/Taylor and automatic-differentiation machinery that the Optimization, Optimal Control, and Machine Learning tracks assume as prerequisites.

## Position in the curriculum

- Parent track: `01_Foundations` (follows `01_Linear_Algebra`).
- Section numbering: `1.3.<subfolder>.<notebook>` (e.g. the first notebook is `1.3.1.1`), mirroring the folder hierarchy exactly as the Linear Algebra track does.
- Entry point: `01_Limits_and_Continuity/01_limit_of_a_function.ipynb` (`1.3.1.1`). When that notebook is created, update the previous/next navigation of the last Linear Algebra notebook (`01_Linear_Algebra/09_Multilinear_Algebra_and_Tensors/13_vectors_and_one_forms.ipynb`) to add a `Next` link into Calculus.
- Hand-off: the final subfolder (`07_Matrix_Calculus_and_Automatic_Differentiation`) bridges directly into `03_Optimization` and `05_Machine_Learning`.

## Design principle

Ordered **foundations → complex/specific**: single-variable limits and derivatives first, then integration and series, then multivariable differential calculus (the core optimization/ML prerequisite), then a lean block of multivariable integration and field calculus, and finally matrix calculus and automatic differentiation as the most applied, repository-specific capstone.

## Scope

**In scope**

- Limits, continuity, and the analytical results that guarantee roots and minima exist.
- Single-variable differentiation and integration, including the numerical and differentiation-under-the-integral tools used downstream.
- Sequences, series, and Taylor approximation (single- and multivariable).
- Multivariable differential calculus: partial derivatives, gradient, directional derivative, Jacobian, multivariable chain rule, Hessian, second-order Taylor expansion, the implicit function theorem.
- A **lean** block of multivariable integration and vector (field) calculus: iterated/triple integrals, the change-of-variables Jacobian determinant, gradient fields, divergence, the Laplacian, and the divergence theorem (integration by parts in space).
- Matrix calculus and automatic differentiation (forward and reverse mode / backpropagation).

**Out of scope — handled by dedicated tracks, do not duplicate here**

- Differential equations (ODE/PDE solution methods, systems, numerical integrators) → `01_Foundations/05_Differential_Equations`. This track contains **no** differential-equation content; it only supplies the derivative-as-rate-of-change foundation those topics build on.
- Calculus of variations / Euler–Lagrange / functionals → `01_Foundations/04_Calculus_of_Variations`.
- Integral transforms (Laplace, Fourier, Z) → `01_Foundations/06_Integral_Transforms`.
- Probability, expectation, and measure machinery → `01_Foundations/02_Probability_and_Statistics`.
- Curl, line/surface integrals, Green's and Stokes' theorems — excluded as outside the repository's learning goals (this ML/control/RL curriculum never consumes them).

---

## Subfolder architecture

| # | Subfolder | Notebooks | Primary downstream consumers |
|---|-----------|-----------|------------------------------|
| 1 | `01_Limits_and_Continuity` | 7 | Existence of optima, convergence, root-bracketing |
| 2 | `02_Single_Variable_Differentiation` | 11 | Optimality conditions, line search, ML activations/backprop |
| 3 | `03_Single_Variable_Integration` | 8 | Cost functionals, expectations, quadrature, policy gradients |
| 4 | `04_Sequences_Series_and_Taylor_Approximation` | 6 | Second-order methods, error bounds, model linearization |
| 5 | `05_Multivariable_Differential_Calculus` | 12 | **Core**: gradients/Jacobians/Hessians for optimization, control, ML |
| 6 | `06_Multivariable_Integration_and_Vector_Calculus` | 7 | Change-of-variables densities, PDE-constrained control, transport modeling |
| 7 | `07_Matrix_Calculus_and_Automatic_Differentiation` | 9 | Backpropagation, CasADi AD, gradients of quadratic forms |

**Total: 60 notebooks across 7 subfolders.**

---

### 1.3.1 — `01_Limits_and_Continuity`

The analytical foundation derivatives rest on, plus the existence results the Optimization track invokes.

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.1.1 | `01_limit_of_a_function` | Limit of a function as the approaching value of `f`. |
| 1.3.1.2 | `02_limit_laws` | Algebraic rules for evaluating limits. |
| 1.3.1.3 | `03_one_sided_limits_and_limits_at_infinity` | One-sided and asymptotic limits. |
| 1.3.1.4 | `04_continuity` | Continuity at a point and on an interval. |
| 1.3.1.5 | `05_asymptotic_notation_big_o_little_o` | `O(·)` / `o(·)` growth and error notation → Taylor remainder, optimization convergence rates. |
| 1.3.1.6 | `06_intermediate_value_theorem` | Guaranteed roots of continuous functions → bisection / bracketing in line search. |
| 1.3.1.7 | `07_extreme_value_theorem` | Existence of minima/maxima on compact sets (Weierstrass) → existence of optima. |

### 1.3.2 — `02_Single_Variable_Differentiation`

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.2.1 | `01_derivative_definition` | Difference quotient and the tangent line. |
| 1.3.2.2 | `02_differentiation_rules` | Sum, product, and quotient rules. |
| 1.3.2.3 | `03_chain_rule` | Derivative of a composition → backpropagation. |
| 1.3.2.4 | `04_derivatives_of_elementary_functions` | `exp`, `log`, trigonometric derivatives. |
| 1.3.2.5 | `05_derivatives_of_activation_functions` | Sigmoid, tanh, softplus, ReLU → ML gradients. |
| 1.3.2.6 | `06_higher_order_derivatives` | Second and higher-order derivatives. |
| 1.3.2.7 | `07_mean_value_theorem` | MVT → descent-lemma and convergence proofs. |
| 1.3.2.8 | `08_monotonicity_and_first_derivative_test` | Increasing/decreasing behavior, local extrema. |
| 1.3.2.9 | `09_convexity_and_second_derivative_test` | Convexity/concavity and inflection → sufficient optimality conditions. |
| 1.3.2.10 | `10_linear_approximation` | First-order (tangent-line) approximation. |
| 1.3.2.11 | `11_newtons_method_for_root_finding` | Root-finding by linearization (the calculus primitive, distinct from Newton-for-minimization in Optimization). |

### 1.3.3 — `03_Single_Variable_Integration`

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.3.1 | `01_antiderivative_and_indefinite_integral` | Antiderivatives. |
| 1.3.3.2 | `02_definite_integral_and_riemann_sums` | The definite integral as a limit of Riemann sums. |
| 1.3.3.3 | `03_fundamental_theorem_of_calculus` | Links differentiation and integration. |
| 1.3.3.4 | `04_integration_by_substitution` | Substitution rule. |
| 1.3.3.5 | `05_integration_by_parts` | Integration by parts → adjoint/Leibniz manipulations. |
| 1.3.3.6 | `06_improper_integrals` | Unbounded domains/integrands → normalizers and expectations. |
| 1.3.3.7 | `07_numerical_integration` | Trapezoidal and Simpson quadrature → cost integrals, collocation, returns. |
| 1.3.3.8 | `08_differentiation_under_the_integral_sign` | Leibniz rule → policy gradients, reparameterization, sensitivity. |

### 1.3.4 — `04_Sequences_Series_and_Taylor_Approximation`

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.4.1 | `01_sequences_and_convergence` | Sequences and their limits. |
| 1.3.4.2 | `02_infinite_series_and_partial_sums` | Series and convergence of partial sums. |
| 1.3.4.3 | `03_power_series` | Power series and radius of convergence. |
| 1.3.4.4 | `04_taylor_and_maclaurin_series` | Taylor/Maclaurin expansions. |
| 1.3.4.5 | `05_taylors_theorem_with_remainder` | Remainder bounds → approximation-error control. |
| 1.3.4.6 | `06_quadratic_approximation_and_curvature` | Second-order single-variable model → bridges to multivariable second-order methods. |

### 1.3.5 — `05_Multivariable_Differential_Calculus`

The core prerequisite block for Optimization, Optimal Control, and Machine Learning.

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.5.1 | `01_functions_of_several_variables` | Scalar fields, graphs, and level sets. |
| 1.3.5.2 | `02_partial_derivatives` | Partial derivatives. |
| 1.3.5.3 | `03_gradient` | The gradient vector → gradient descent. |
| 1.3.5.4 | `04_directional_derivative` | Rate of change along a direction. |
| 1.3.5.5 | `05_gradient_and_level_sets` | Gradient ⟂ level set; steepest ascent/descent geometry. |
| 1.3.5.6 | `06_linearization_and_tangent_plane` | First-order multivariable approximation → model linearization in control. |
| 1.3.5.7 | `07_jacobian` | Derivative of vector-valued maps → Gauss–Newton, state-space linearization. |
| 1.3.5.8 | `08_multivariable_chain_rule` | Composition derivative → backpropagation, adjoint method. |
| 1.3.5.9 | `09_hessian_matrix` | Second-order partials → Newton's method, curvature. |
| 1.3.5.10 | `10_second_order_taylor_expansion` | Multivariable second-order expansion → optimality conditions, trust regions. |
| 1.3.5.11 | `11_critical_points_and_second_derivative_test` | Definiteness, saddle points → necessary/sufficient optimality. |
| 1.3.5.12 | `12_implicit_function_theorem` | Solvability and sensitivity → KKT, parametric sensitivity. |

### 1.3.6 — `06_Multivariable_Integration_and_Vector_Calculus`

Lean field calculus: only the operators consumed by PDE-constrained optimal control, transport/process modeling, and change-of-variables in probability/ML.

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.6.1 | `01_double_and_iterated_integrals` | Iterated integration over regions. |
| 1.3.6.2 | `02_triple_integrals` | Integration over volumes. |
| 1.3.6.3 | `03_change_of_variables_and_jacobian_determinant` | Jacobian-determinant volume scaling → probability densities, normalizing flows. |
| 1.3.6.4 | `04_vector_fields_and_gradient_fields` | Vector fields and conservative (gradient) fields / potentials. |
| 1.3.6.5 | `05_divergence` | Divergence operator → conservation/transport laws. |
| 1.3.6.6 | `06_laplacian` | The Laplacian → diffusion, Fokker–Planck, regularization. |
| 1.3.6.7 | `07_divergence_theorem_and_integration_by_parts` | Gauss/divergence theorem and integration by parts in space → adjoint PDE-constrained control. |

### 1.3.7 — `07_Matrix_Calculus_and_Automatic_Differentiation`

The most applied, repository-specific capstone: the derivative machinery the CasADi (AD) and TensorFlow (backprop) stack is built on. Hands off into `03_Optimization` and `05_Machine_Learning`.

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.3.7.1 | `01_derivatives_with_respect_to_a_vector` | Vector-by-vector derivatives and layout conventions. |
| 1.3.7.2 | `02_derivatives_with_respect_to_a_matrix` | Scalar/matrix-by-matrix derivatives. |
| 1.3.7.3 | `03_gradients_of_linear_and_quadratic_forms` | `∇(aᵀx)`, `∇(xᵀQx)` → quadratic objectives, LQR cost. |
| 1.3.7.4 | `04_derivatives_of_trace_determinant_and_log_det` | Trace/determinant/log-det gradients → Gaussian likelihoods, log-det barriers. |
| 1.3.7.5 | `05_matrix_chain_rule` | Chaining matrix-valued derivatives. |
| 1.3.7.6 | `06_jacobian_vector_and_vector_jacobian_products` | JVP and VJP → forward/reverse AD primitives. |
| 1.3.7.7 | `07_forward_mode_automatic_differentiation` | Dual-number / forward-accumulation AD. |
| 1.3.7.8 | `08_reverse_mode_automatic_differentiation` | Backpropagation. |
| 1.3.7.9 | `09_numerical_symbolic_and_automatic_differentiation` | Finite-difference vs symbolic vs AD; gradient checking. |

---

## Literature mapping

Primary canonical source for the track:

- **Aazi, M. (2024). *Mathematics For Machine Learning* — Chapter 3, Calculus.** Maps to subfolders 1.3.1–1.3.6. (The in-repo folder `Literature/01 - Foundations/Aazi 2024 - Mathematics For Machine Learning/Chapter03_Calculus` is currently an empty placeholder — the chapter still needs converting before notebook authoring, per ARCH-RULES §9.)

Supplementary calculus rigor (already in `Literature/`, optimization-oriented):

- **Bertsekas, D. P. (1999). *Nonlinear Programming* — Appendix A.5 (Derivatives).** Gradient, Jacobian, Hessian, multivariable chain rule, mean value theorem, second-order (Taylor) expansion, implicit function theorem. Maps to subfolders 1.3.5 and 1.3.7.
- **Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization* — Appendix A.4 (Derivatives).** Derivative/Jacobian, gradient, first- and second-order approximation, chain rule, Hessian, matrix-calculus examples (quadratic form, `log det`). Maps to subfolders 1.3.5 and 1.3.7.

Per ARCH-RULES §10, notebook explanations stay author-neutral; books appear only in the Cell 8 references with public URLs (Aazi MML, `athenasc.com` for Bertsekas, `web.stanford.edu/~boyd/cvxbook` for Boyd), never as local `Literature/` links.

## Computational stack

Follow CODING-RULES §8 (SymPy-first, then a domain-centric equivalent). For this track specifically:

- **SymPy first**, always: `sp.limit`, `sp.diff`, `sp.integrate`, `sp.series`, `Matrix.jacobian`, `sp.hessian` produce the exact, textbook-style intermediate steps.
- **CasADi equivalent** in any cell that *evaluates a derivative* (gradient/Jacobian/Hessian/partial of an explicit function), per the established `03_Optimization` pattern: `ca.gradient` / `ca.jacobian` / `ca.hessian`, with `ca.substitute`/`ca.evalf` to match the SymPy numbers. The `07_Matrix_Calculus_and_Automatic_Differentiation` subfolder is where CasADi forward/reverse AD is the natural domain workflow.
- **TensorFlow equivalent** for the automatic-differentiation notebooks (`1.3.7.7`–`1.3.7.9`): `tf.GradientTape` for reverse-mode/backprop, `tf.autodiff.ForwardAccumulator` for forward-mode.
- **NumPy / SciPy / Matplotlib** are supporting tools only — quadrature checks (`07_numerical_integration`), finite-difference comparisons, and any visualization (e.g. tangent lines, level sets, gradient fields). They must not replace SymPy for the from-scratch computation.

## Notebook structure reminder

Every notebook is the mandatory 8-cell layout (ARCH-RULES §8.1–8.2). As a Foundations track (ARCH-RULES §8.4): keep examples small and numerical, emphasize that the algebraic and coded results match, and include the optional visualization cell only when geometry genuinely aids intuition (tangent line, level sets and gradient direction, a vector field, a Taylor-approximation error plot). Cross-reference related concepts naturally per ARCH-RULES §10 — e.g. the gradient links back to the Linear Algebra notebooks on inner products and the Jacobian to `matrix_vector_multiplication`.
