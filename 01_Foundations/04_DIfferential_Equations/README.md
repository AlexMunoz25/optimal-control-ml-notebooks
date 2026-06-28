# 04_Differential_Equations — Track Architecture

Differential-equations foundation track for `01_Foundations`. This file is the canonical plan for the track: it fixes the subfolders, the notebook sequence, the section numbering, the literature mapping, and the computational stack so future agents can author conforming notebooks without re-deriving the design.

> **Status:** complete — all 40 notebooks authored against the canonical source (Zill, *A First Course in Differential Equations with Modeling Applications*), executed, and validated. The architecture below is the canonical record of the track.

Authoring rules are **not** restated here. Every notebook must follow the root guidance exactly:
[AGENTS.md](../../AGENTS.md) · [CONTEXT.md](../../CONTEXT.md) · [CODING-RULES.md](../../CODING-RULES.md) · [ARCH-RULES.md](../../ARCH-RULES.md) (mandatory 8-cell structure, §8.4 *Foundations* rules apply).

---

## Purpose

Provide exactly the ordinary-differential-equation theory and solution methods that the repository's later domains depend on — process/dynamics modeling, optimal control, machine learning, and reinforcement learning — and nothing beyond those learning goals. The track builds from the meaning of a differential equation up to the linear-systems, matrix-exponential, Laplace-transform, and numerical-integration machinery that the Dynamics, Optimal Control, and RL tracks assume as prerequisites.

## Position in the curriculum

- Parent track: `01_Foundations` (follows `03_Calculus`; differential equations are the rate-of-change models that calculus's derivative and integral make precise).
- Section numbering: `1.4.<subfolder>.<notebook>` (e.g. the first notebook is `1.4.1.1`), mirroring the folder hierarchy exactly as the Calculus track does.
- Entry point: `01_Introduction/01_definitions_and_terminology.ipynb` (`1.4.1.1`).
- Hand-off: the final subfolders (`08_Systems_of_Linear_First_Order_ODEs`, `09_Numerical_Methods`) bridge directly into `02_Dynamics` and `04_Optimal_Control` (state-space models, $e^{At}$, simulation).

## Design principle

Ordered **by type and method, simplest → advanced**, not by chapter: first the meaning of an ODE and initial-value problems, then every first-order solution method, then first-order modeling, then higher-order linear theory and its solution methods, then higher-order modeling, then the Laplace-transform operational calculus, then series solutions, then linear systems in matrix form, and finally numerical integration as the most applied, repository-facing capstone.

## Scope

**In scope**

- Meaning, classification (type/order/linearity), solutions, initial-value problems, and the existence/uniqueness theorem.
- All elementary first-order methods: direction fields, autonomous equations and stability, separable, linear, exact, integrating factors, homogeneous (substitution), Bernoulli.
- First-order modeling: growth/decay, Newton's law of cooling, mixtures, logistic population.
- Higher-order linear theory (superposition, Wronskian, general solution) and methods: reduction of order, constant-coefficient homogeneous, undetermined coefficients, variation of parameters, Cauchy–Euler.
- Higher-order modeling: spring/mass systems, driven motion and resonance, LRC circuits (the canonical second-order dynamics the control track linearizes).
- The Laplace transform as an operational method for IVPs: definition, inverse, transforms of derivatives, translation theorems and step functions, convolution, the Dirac delta, and the transfer-function / impulse-response viewpoint.
- Series solutions about ordinary points and the method of Frobenius.
- Systems of linear first-order ODEs in matrix form: the eigenvalue method, repeated/complex eigenvalues, and the matrix exponential $e^{At}$ (the LTI state-space solution).
- Numerical methods: Euler, improved Euler, Runge–Kutta, and their extension to systems and higher-order equations.

**Out of scope — handled by dedicated tracks or beyond the repository's goals, do not duplicate here**

- Integral-transform *theory* (Fourier, Z, and the general transform machinery) → `01_Foundations/06_Integral_Transforms`. The Laplace transform appears here only as an ODE/IVP solution method, per Zill Chapter 7.
- Partial differential equations and boundary-value problems beyond the brief mention needed for context.
- Special functions (Bessel, Legendre) and the annihilator, Green's-function, and elimination variants — alternative machinery outside the repository's learning goals.
- Calculus of variations / Euler–Lagrange → `01_Foundations/05_Calculus_of_Variations`.

---

## Subfolder architecture

| # | Subfolder | Notebooks | Primary downstream consumers |
|---|-----------|-----------|------------------------------|
| 1 | `01_Introduction` | 4 | Vocabulary and well-posedness for every later model |
| 2 | `02_First_Order_Equations` | 8 | First-order dynamics, stability of equilibria |
| 3 | `03_Modeling_with_First_Order_Equations` | 4 | Process modeling, RL/control environments |
| 4 | `04_Higher_Order_Linear_Equations` | 6 | Linear dynamics, characteristic roots, stability |
| 5 | `05_Modeling_with_Higher_Order_Equations` | 3 | Mechanical/electrical plants for control |
| 6 | `06_Laplace_Transforms` | 6 | Transfer functions, impulse response, control design |
| 7 | `07_Series_Solutions` | 2 | Approximate/local solutions, special models |
| 8 | `08_Systems_of_Linear_First_Order_ODEs` | 4 | **Core**: state-space $x' = Ax$, $e^{At}$, stability |
| 9 | `09_Numerical_Methods` | 3 | **Core**: simulation, collocation, neural ODEs |

**Total: 40 notebooks across 9 subfolders.**

---

### 1.4.1 — `01_Introduction`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.1.1 | `01_definitions_and_terminology` | DE, classification by type/order/linearity, normal form. |
| 1.4.1.2 | `02_initial_value_problems` | Solutions, particular vs general, the IVP. |
| 1.4.1.3 | `03_differential_equations_as_mathematical_models` | How rate laws become ODEs. |
| 1.4.1.4 | `04_existence_and_uniqueness_of_solutions` | Theorem 1.2.1 (continuity of $f$, $\partial f/\partial y$). |

### 1.4.2 — `02_First_Order_Equations`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.2.1 | `01_direction_fields` | Slope fields and qualitative solution curves. |
| 1.4.2.2 | `02_autonomous_equations_and_stability` | Critical points, phase line, attractors/repellers. |
| 1.4.2.3 | `03_separable_equations` | Separation of variables. |
| 1.4.2.4 | `04_linear_first_order_equations` | Integrating-factor method for linear ODEs. |
| 1.4.2.5 | `05_exact_equations` | Exactness test and potential function. |
| 1.4.2.6 | `06_integrating_factors` | Special integrating factors for non-exact equations. |
| 1.4.2.7 | `07_homogeneous_equations` | Homogeneous coefficients, substitution $y = ux$. |
| 1.4.2.8 | `08_bernoulli_equations` | Bernoulli substitution $u = y^{1-n}$. |

### 1.4.3 — `03_Modeling_with_First_Order_Equations`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.3.1 | `01_growth_and_decay` | Exponential growth/decay, half-life. |
| 1.4.3.2 | `02_newtons_law_of_cooling` | Cooling/warming toward ambient. |
| 1.4.3.3 | `03_mixtures` | Concentration in a stirred tank. |
| 1.4.3.4 | `04_logistic_population_model` | Logistic growth and carrying capacity. |

### 1.4.4 — `04_Higher_Order_Linear_Equations`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.4.1 | `01_preliminary_theory_of_linear_equations` | Superposition, Wronskian, $y = y_c + y_p$. |
| 1.4.4.2 | `02_reduction_of_order` | Second solution from a known one. |
| 1.4.4.3 | `03_homogeneous_equations_with_constant_coefficients` | Auxiliary equation, real/repeated/complex roots. |
| 1.4.4.4 | `04_undetermined_coefficients` | Particular solution by trial forms. |
| 1.4.4.5 | `05_variation_of_parameters` | Particular solution for general forcing. |
| 1.4.4.6 | `06_cauchy_euler_equations` | Variable-coefficient $x^k y^{(k)}$ equations. |

### 1.4.5 — `05_Modeling_with_Higher_Order_Equations`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.5.1 | `01_free_spring_mass_systems` | Simple harmonic and damped free motion. |
| 1.4.5.2 | `02_driven_motion_and_resonance` | Forced oscillation, beats, resonance. |
| 1.4.5.3 | `03_lrc_series_circuits` | The electrical second-order analogue. |

### 1.4.6 — `06_Laplace_Transforms`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.6.1 | `01_definition_of_the_laplace_transform` | $\mathcal{L}\{f\}$ as an integral operator. |
| 1.4.6.2 | `02_inverse_laplace_transform` | Partial fractions and the inverse. |
| 1.4.6.3 | `03_solving_initial_value_problems` | Transform of derivatives → algebra → invert. |
| 1.4.6.4 | `04_translation_theorems_and_step_functions` | Shifting theorems, unit step, piecewise forcing. |
| 1.4.6.5 | `05_convolution_theorem` | Products of transforms, transforms of integrals. |
| 1.4.6.6 | `06_dirac_delta_and_transfer_functions` | Impulse response and the transfer function. |

### 1.4.7 — `07_Series_Solutions`

| Section | Notebook | Topic |
|---------|----------|-------|
| 1.4.7.1 | `01_power_series_solutions_about_ordinary_points` | Recurrence relations from power series. |
| 1.4.7.2 | `02_method_of_frobenius` | Solutions about regular singular points. |

### 1.4.8 — `08_Systems_of_Linear_First_Order_ODEs`

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.4.8.1 | `01_linear_systems_in_matrix_form` | $\mathbf{x}' = A\mathbf{x}$, the state-space form. |
| 1.4.8.2 | `02_homogeneous_systems_eigenvalue_method` | Distinct real eigenvalues. |
| 1.4.8.3 | `03_repeated_and_complex_eigenvalues` | Generalized eigenvectors, oscillatory modes. |
| 1.4.8.4 | `04_matrix_exponential` | $e^{At}$ as the LTI solution operator. |

### 1.4.9 — `09_Numerical_Methods`

| Section | Notebook | Topic & downstream use |
|---------|----------|------------------------|
| 1.4.9.1 | `01_eulers_method` | The first-order step and local/global error. |
| 1.4.9.2 | `02_runge_kutta_methods` | Improved Euler and classical RK4. |
| 1.4.9.3 | `03_numerical_methods_for_systems_and_higher_order` | Vectorized stepping for systems and IVPs. |

---

## Literature mapping

Primary canonical source for the track — **Zill, D. G. *A First Course in Differential Equations with Modeling Applications*.** The notebooks preserve its terminology, notation, definitions, theorems, modeling context, and worked examples, adapting only the order to this curriculum (by type/method rather than chapter). Chapter → subfolder mapping:

| Subfolder | Zill chapter(s) |
|-----------|-----------------|
| 1.4.1 `Introduction` | Ch 1 (Definitions, IVPs, models, existence/uniqueness) |
| 1.4.2 `First_Order_Equations` | Ch 2 (2.1 direction fields/autonomous, 2.2 separable, 2.3 linear, 2.4 exact, 2.5 substitutions) |
| 1.4.3 `Modeling_with_First_Order_Equations` | Ch 3 (3.1 linear models, 3.2 nonlinear/logistic) |
| 1.4.4 `Higher_Order_Linear_Equations` | Ch 4 (4.1 theory, 4.2 reduction of order, 4.3 constant coeff, 4.4 undetermined coeff, 4.6 variation of parameters, 4.7 Cauchy–Euler) |
| 1.4.5 `Modeling_with_Higher_Order_Equations` | Ch 5 (5.1 spring/mass, driven motion, LRC circuits) |
| 1.4.6 `Laplace_Transforms` | Ch 7 (7.1–7.5) |
| 1.4.7 `Series_Solutions` | Ch 6 (6.2 ordinary points, 6.3 Frobenius) |
| 1.4.8 `Systems_of_Linear_First_Order_ODEs` | Ch 8 (8.1 theory, 8.2 eigenvalue method, 8.4 matrix exponential) |
| 1.4.9 `Numerical_Methods` | Ch 9 (9.1 Euler, 9.2 Runge–Kutta, 9.4 systems/higher-order) |

Per ARCH-RULES §10, notebook explanations stay author-neutral; the book appears only in the Cell 8 references with a public URL (the Cengage product page), never as a local `Literature/` link.

## Computational stack

Follow CODING-RULES §8 (SymPy-first, then a domain-centric equivalent). For this track specifically:

- **SymPy first**, always: `sp.dsolve`, `sp.classify_ode`, `sp.laplace_transform` / `sp.inverse_laplace_transform`, `sp.series`, `Matrix.eigenvects`, `(A*t).exp()` produce the exact, textbook-style intermediate steps. Every solving example is worked by hand in the markdown first, then mirrored in SymPy with printed intermediate values.
- **SciPy / CasADi equivalent** for the numerical-methods and modeling notebooks: `scipy.integrate.solve_ivp` and CasADi integrators (`ca.integrator`) are the natural numerical workflow for simulating an IVP, and CasADi is the repository's control-oriented integrator. From-scratch Euler/RK steppers come first (transparent), then the library integrator as the equivalent.
- **NumPy / Matplotlib** are supporting tools only — direction fields, phase lines, solution and error plots. They must not replace SymPy for the from-scratch computation.

## Notebook structure reminder

Every notebook is the mandatory 8-cell layout (ARCH-RULES §8.1–8.2). As a Foundations track (ARCH-RULES §8.4): keep examples small and explicit, emphasize that the hand computation and the coded result match, and include the optional visualization cell only when geometry genuinely aids intuition (direction field, phase line, solution curve, resonance envelope, Euler-vs-exact error). **Every solving example must be fully step by step** — show each algebraic move, substitution, integration, constant of integration, application of initial conditions, and interpretation of the final solution. Cross-reference related concepts naturally per ARCH-RULES §10 — e.g. separable equations link back to [integration by substitution](../03_Calculus/03_Single_Variable_Integration/04_integration_by_substitution.ipynb), the Laplace transform to [improper integrals](../03_Calculus/03_Single_Variable_Integration/06_improper_integrals.ipynb), series solutions to [power series](../03_Calculus/04_Sequences_Series_and_Taylor_Approximation/03_power_series.ipynb), and linear systems to [eigenvalues](../01_Linear_Algebra/07_Theoretical_Linear_Algebra/08_eigenvalue.ipynb) and the [matrix exponential / eigendecomposition](../01_Linear_Algebra/07_Theoretical_Linear_Algebra/12_eigendecomposition.ipynb).
