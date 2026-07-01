# Optimal Control and Machine Learning — Jupyter Learning Hub

Notebook learning path for mathematics and computation behind modern control, optimization, machine learning, and reinforcement learning.

## About Me

Hi, I am a software and simulation engineer with about 9 years of experience across aerospace student competitions, industrial simulation software, and R&D engineering. I focus on optimal control, RTO, ML, and dynamic-system modeling, I currently serve as senior R&D in robotics company, and CTO of a startup.

## Why This Repository

I learn best by building. If these notes help someone else along the way, even better.

This project is for readers who want a coherent path from linear algebra and calculus foundations into optimization, optimal control, ML systems, and reinforcement learning. It is especially useful if you want to see how equations become working code, how prerequisite ideas connect across subjects, and how canonical textbook material can be turned into small, executable lessons.

Designed for:

- students building a serious mathematical foundation for control, ML, or robotics
- engineers who want executable refreshers instead of isolated formulas
- self-learners who prefer a sequential and wiki-like path with previous/next navigation
- researchers and practitioners who want compact notebooks grounded in source material

This is not a software framework. Its a learning hub: notebooks are written to teach the calculation, expose intermediate steps, and connect each lesson to others.

## How Notebooks Teach

Each notebook follows same arch:

- a numbered title that matches the folder hierarchy
- a mathematical definition or main equation at the top
- a concise explanation that interprets the symbols and main idea
- figures, visualizations, or extended variants when they help intuition
- a numerical example that works through the calculation step by step
- a Python implementation whose output makes the same reasoning visible
- references to the source material and previous/next navigation links

## Computational Stack

- `sympy` for symbolic algebra, exact arithmetic, derivations, and from-scratch mathematical computation.
- `casadi` for optimization, automatic differentiation, nonlinear programming, and control-oriented numerical optimization.
- `tensorflow` for machine learning sections, differentiable models, training loops, and neural-network examples.
- `python-control` for control or systems topics.
- `numpy`, `scipy`, for plotting or where they support arrays, numerical checks, or visualization.

---

## Repository Structure So Far

```
optimal-control-ml-notebooks/
│
├── README.md
├── requirements.txt
│
├── 01_Foundations/                                           ← 123 notebooks
│   ├── 01_Linear_Algebra/                                    ← 123 notebooks
│   │   ├── 01_Prerequisites/                                 (3 notebooks)
│   │   ├── 02_Vector/                                        (9 notebooks)
│   │   ├── 03_Matrix/                                        (20 notebooks)
│   │   ├── 04_Computational_Linear_Algebra/                  (6 notebooks)
│   │   ├── 05_Geometrical_Aspects_of_Linear_Algebra/         (17 notebooks)
│   │   ├── 06_Linear_Transformations/                        (14 notebooks)
│   │   ├── 07_Theoretical_Linear_Algebra/                    (31 notebooks)
│   │   ├── 08_Coordinate_Transformations/                    (10 notebooks)
│   │   └── 09_Multilinear_Algebra_and_Tensors/               (13 notebooks)
│   ├── 02_Probability_and_Statistics/                        ← planned
│   ├── 03_Calculus/                                          (60 notebooks)
│   │   ├── 01_Limits_and_Continuity/                         (7 notebooks)
│   │   ├── 02_Single_Variable_Differentiation/              (11 notebooks)
│   │   ├── 03_Single_Variable_Integration/                   (8 notebooks)
│   │   ├── 04_Sequences_Series_and_Taylor_Approximation/     (6 notebooks)
│   │   ├── 05_Multivariable_Differential_Calculus/          (12 notebooks)
│   │   ├── 06_Multivariable_Integration_and_Vector_Calculus/ (7 notebooks)
│   │   └── 07_Matrix_Calculus_and_Automatic_Differentiation/ (9 notebooks)
│   ├── 04_Calculus_of_Variations/                            ← planned
│   ├── 05_DIfferential_Equations/                            ← planned
│   └── 06_Integral_Transforms/                               ← planned
│
├── 02_Dynamics/                                              ← planned
│   ├── 01_Causal_Acausal_Modeling/                           ← planned
│   ├── 02_Modeling_using_Lagrange/                           ← planned
│   ├── 03_Modeling_using_Bond_Graph/                         ← planned
│   └── 04_Modeling_using_Port_Hamiltonian/                   ← planned
│
├── 03_Optimization/                                          ← 154 notebooks
│   ├── 01_Optimization_Fundamentals/                         (6 notebooks)
│   ├── 02_Continuous_Optimization/                           (6 notebooks)
│   ├── 03_Convex_Optimization/                               (52 notebooks)
│   ├── 04_Convex_Problem_Classes/                            (14 notebooks)
│   ├── 05_Nonlinear_Programming/                             (13 notebooks)
│   ├── 06_Discrete_and_Combinatorial_Optimization/           (6 notebooks)
│   ├── 07_Stochastic_and_Robust_Optimization/                (4 notebooks)
│   ├── 08_Global_and_Derivative_Free_Optimization/           (5 notebooks)
│   └── 09_Algorithms/                                        (48 notebooks)
│
├── 04_Optimal_Control/                                       ← planned
│   ├── 01_Controllability_and_Observability/                 ← planned
│   ├── 02_Kalman_FIlter/                                     ← planned
│   ├── 03_Full_State_Feedback_Control/                       ← planned
│   ├── 04_Linear_Quadratic_Regulator_LQR/                    ← planned
│   ├── 05_Linear_Quadratic_Gaussian_LQG/                     ← planned
│   ├── 06_Trajectory_Optimization_DDP_iLQR/                  ← planned
│   ├── 07_Model_Predictive_Control_MPC/                      ← planned
│   ├── 08_Nonlinear_Model_Predictive_Control_NMPC/           ← planned
│   ├── 09_Robust_and_H_infinity_Control/                     ← planned
│   ├── 10_Hybrid_and_Switched_Systems_Control/               ← planned
│   └── 11_PDE_Constrained_Optimal_Control/                   ← planned
│
├── 05_Machine_Learning/                                      ← planned
├── 07_Reinforcement_Learning/                                ← planned
│
├── A1_OOP/                                                   ← 13 notebooks
│   ├── 01_Object_Oriented_Principle/                         (7 notebooks)
│   └── 02_Object_Oriented_Design_Patterns/                   (6 notebooks)
├── A2_LLVM_Compilers/                                        ← 16 notebooks
│   ├── 01_Compiler_Foundations/                              (4 notebooks)
│   ├── 02_LLVM_Intermediate_Representation/                  (3 notebooks)
│   ├── 03_Pass_Infrastructure/                               (3 notebooks)
│   ├── 04_Optimization_Topics/                               (3 notebooks)
│   └── 05_MLIR/                                              (3 notebooks)
├── A3_Performance_and_Systems/                               ← 9 notebooks
│   ├── 01_Central_Processing_Unit_Performance/               (3 notebooks)
│   ├── 02_Vectorization/                                     (2 notebooks)
│   ├── 03_Parallelism/                                       (2 notebooks)
│   └── 04_Profiling/                                         (2 notebooks)
├── A4_ML_Infrastructure/                                     ← 12 notebooks
│   ├── 01_ML_Compiler_Ecosystem/                             (3 notebooks)
│   ├── 02_Framework_Integration/                             (3 notebooks)
│   ├── 03_Runtime_Topics/                                    (3 notebooks)
│   └── 04_Benchmarking_and_Regressions/                      (3 notebooks)
├── A5_Interview_Katas/                                       ← 28 notebooks
│   ├── 01_Object_Oriented_Programming_Katas/                 (15 notebooks)
│   └── 04_Machine_Learning_Katas/                            (13 notebooks)
│
├── Figures/                                                  ← 150 reusable figure files
└── Literature/                                               ← source material and references
```

---

## Notebook Format

Every notebook follows a consistent lesson architecture:

| Cell | Type     | Content                                                                          |
| ---- | -------- | -------------------------------------------------------------------------------- |
| 1    | Markdown | Section number and title                                                         |
| 2    | Markdown | Core mathematical definition or equation                                         |
| 3    | Markdown | Theoretical explanation and symbol interpretation                                |
| 4    | Markdown | Numerical example with intermediate steps                                        |
| 5    | Code     | `sympy` implementation first, with domain-specific equivalent code when useful |
| 6    | Code     | Optional figures, advanced analysis, or visualization                            |
| 7    | Markdown | Optional extended theory or variants                                             |
| 8    | Markdown | References + Previous / Next links                                               |

---

## Current Progress

| Section                                         | Topic                                                                                                                                | Notebooks | Status       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------- | ------------ |
| 01 Foundations / 01 Linear Algebra              | Prerequisites                                                                                                                        | 3         | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Vectors                                                                                                                              | 9         | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Matrices                                                                                                                             | 20        | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Computational Linear Algebra                                                                                                         | 6         | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Geometrical Aspects                                                                                                                  | 17        | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Linear Transformations                                                                                                               | 14        | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Theoretical Linear Algebra                                                                                                           | 31        | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Coordinate Transformations                                                                                                           | 10        | ✅ Available |
| 01 Foundations / 01 Linear Algebra              | Multilinear Algebra and Tensors                                                                                                      | 13        | ✅ Available |
| 01 Foundations / 03 Calculus                    | Limits, differentiation, integration, series/Taylor, multivariable calculus, vector calculus, matrix calculus and automatic differentiation | 60        | ✅ Available |
| 01 Foundations / 02, 04–06                      | Probability, Variations, Differential Equations, Integral Transforms                                                                  | 0         | 📋 Planned   |
| 02 Dynamics                                     | Causal/Acausal, Lagrange, Bond Graph, Port-Hamiltonian                                                                               | 0         | 📋 Planned   |
| 03 Optimization / 01 Optimization Fundamentals   | Mathematical programming, objective functions, decision variables, feasible sets, local vs global optima, problem classification     | 6         | ✅ Available |
| 03 Optimization / 02 Continuous Optimization     | Unconstrained, constrained, smooth and nonsmooth optimization, nonlinear programming overview                                        | 6         | ✅ Available |
| 03 Optimization / 03 Convex Optimization         | Convex sets, convex functions, convex problems, duality, KKT conditions, interior-point methods                                      | 52        | ✅ Available |
| 03 Optimization / 04 Convex Problem Classes      | LP, QP, QCQP, SOCP, SDP, geometric and conic programming, applied convex programs                                                    | 14        | ✅ Available |
| 03 Optimization / 05 Nonlinear Programming       | General/convex/nonconvex NLP, sequential quadratic programming, augmented Lagrangian, penalty and barrier methods                    | 13        | ✅ Available |
| 03 Optimization / 06 Discrete & Combinatorial    | Integer, binary, mixed-integer linear/quadratic/nonlinear programming, combinatorial optimization                                    | 6         | ✅ Available |
| 03 Optimization / 07 Stochastic & Robust         | Stochastic programming, chance-constrained, robust and distributionally robust optimization                                          | 4         | ✅ Available |
| 03 Optimization / 08 Global & Derivative-Free    | Global optimization, branch and bound, Bayesian optimization, evolutionary and derivative-free methods                               | 5         | ✅ Available |
| 03 Optimization / 09 Algorithms                  | Gradient descent, Newton, quasi-Newton, conjugate gradient, active-set, interior-point, first-order methods                          | 48        | ✅ Available |
| 04 Optimal Control                              | Controllability, estimation, LQR, LQG, trajectory optimization, MPC, robust control, hybrid systems, PDE-constrained optimal control | 0         | 📋 Planned   |
| 05 Machine Learning                             | Core modules                                                                                                                         | 0         | 📋 Planned   |
| 07 Reinforcement Learning                       | Core modules                                                                                                                         | 0         | 📋 Planned   |
| A1 OOP                                          | Object-oriented principles and design patterns                                                                                       | 13        | ✅ Available |
| A2 LLVM Compilers                               | Compiler foundations, LLVM IR, passes, optimization, MLIR                                                                            | 16        | ✅ Available |
| A3 Performance and Systems                      | CPU performance, vectorization, parallelism, profiling                                                                               | 9         | ✅ Available |
| A4 ML Infrastructure                            | ML compilers, framework integration, runtime topics, benchmarking                                                                    | 12        | ✅ Available |
| A5 Interview Katas                              | OOP and machine-learning infrastructure katas                                                                                        | 28        | ✅ Available |

**Current Total: 415 notebooks**

---

# Getting Started

> [!IMPORTANT]
> **You don't need to install anything to explore this hub.**
> All notebooks render directly on GitHub — Just **leave a Star** and enjoy the ride! 🚀

> The setup guide below is **only** for those who want to **run the Python code**, **modify notebooks**, or **experiment locally**. Assumes **no prior developer setup** and walks through everything from scratch.

## 1. Install Git

Git is required to download (clone) the repository.

### macOS

1. Open Terminal
2. Run:

```bash
git --version
```

If Git is not installed, install via:

```bash
xcode-select --install
```

### Windows

1. Go to: https://git-scm.com/download/win
2. Download and install with default settings
3. Restart terminal after install

### Linux (Ubuntu)

```bash
sudo apt update
sudo apt install git
```

Verify:

```bash
git --version
```

## 2. Install Python

Python 3.10 or newer is recommended.

Download from:
https://www.python.org/downloads/

During installation on Windows:
✔ Check **"Add Python to PATH"**

Verify installation:

```bash
python --version
```

or

```bash
python3 --version
```

## 3. Clone the Repository

Open a terminal (Terminal / PowerShell / Command Prompt).

Choose where you want the project folder, then run:

```bash
git clone https://github.com/AlexMunoz25/optimal-control-ml-notebooks.git
cd optimal-control-ml-notebooks
```

This downloads the repo and moves into it.

## 4. Create a Virtual Environment

A virtual environment keeps dependencies isolated.

```bash
python -m venv .venv
```

### Activate it

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` in your terminal.

## 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Install VS Code

Download:
https://code.visualstudio.com/

Install normally.

## 7. Install VS Code Extensions

Open VS Code → Extensions tab → install:

- Python (Microsoft)
- Jupyter (Microsoft)

Or install from terminal:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

## 8. Open the Project

From inside the repo folder:

```bash
code .
```

Or open VS Code → File → Open Folder → select the repo folder.

## 9. Select Python Interpreter

Top-right corner in VS Code:
Select interpreter → choose:

```
.venv
```

## 10. Run Notebooks

Open any `.ipynb` file and press:

- **Run All**
- or run cells individually

VS Code will automatically use the environment.

## Alternative Method — Classic Jupyter

If you prefer standard Jupyter Notebook or Anaconda, follow below.

### Option A — Using pip

```bash
git clone https://github.com/AlexMunoz25/optimal-control-ml-notebooks.git
cd optimal-control-ml-notebooks

python -m venv .venv
source .venv/bin/activate   # Windows equivalent if needed

pip install -r requirements.txt
pip install jupyter

jupyter notebook
```

Browser will open automatically.

### Option B — Using Anaconda

Install Anaconda:
https://www.anaconda.com/download

Then:

```bash
git clone https://github.com/AlexMunoz25/optimal-control-ml-notebooks.git
cd optimal-control-ml-notebooks

conda create -n ocml python=3.11
conda activate ocml

pip install -r requirements.txt
jupyter notebook
```

---

# Updating the Repo

To pull latest changes later:

```bash
git pull
```

## Deactivate Environment

When finished:

```bash
deactivate
```

### (Additional) Tutorial

VS Code + Jupyter setup walkthrough:

https://www.youtube.com/watch?v=9FZzw9nF8Rg

---

## References so far

- Savov, I. (2016). *No Bullshit Guide to Linear Algebra*
- Fleisch, D. (2012). *A Student's Guide to Vectors and Tensors*
- Aazi, M. (2024). *Mathematics For Machine Learning*
- Rozycki, P. (2020). *Computational Mechanics Course Notes, École Centrale de Nantes*
- Bertsekas, D. P. (1999). *Nonlinear Programming*
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*
