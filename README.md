# Optimal Control and Machine Learning — Jupyter Learning Hub

Control and ML notebooks, covering mathematical foundations, dynamic systems, optimization, optimal control, and machine learning.

## Why This Repository

I learn best by building things. If these notes help someone else along the way, even better ...

---

## Repository Structure So Far

```
optimal-control-ml-notebooks/
│
├── README.md
├── AGENTS.md
├── CONTEXT.md
├── CODING-RULES.md
├── ARCH-RULES.md
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
│   ├── 03_Calculus/                                          ← planned
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
├── 03_Optimization/                                          ← 153 notebooks
│   ├── 01_Unconstrained_Optimization/                        (30 notebooks)
│   ├── 02_Convex_Optimization/                               (106 notebooks)
│   │   ├── 01_Introduction/                                  (4 notebooks)
│   │   ├── 02_Convex_Sets/                                   (11 notebooks)
│   │   ├── 03_Convex_Functions/                              (12 notebooks)
│   │   ├── 04_Convex_Optimization_Problems/                  (12 notebooks)
│   │   ├── 05_Duality/                                       (10 notebooks)
│   │   ├── 06_Approximation_and_Fitting/                     (10 notebooks)
│   │   ├── 07_Statistical_Estimation/                        (10 notebooks)
│   │   ├── 08_Geometric_Problems/                            (10 notebooks)
│   │   ├── 09_Unconstrained_Minimization/                    (8 notebooks)
│   │   ├── 10_Equality_Constrained_Minimization/             (8 notebooks)
│   │   └── 11_Interior_Point_Methods/                        (11 notebooks)
│   ├── 03_Lagrange_Multiplier_Theory/                        (17 notebooks)
│   ├── 04_Lagrange_Multiplier_Algorithms/                    ← planned
│   ├── 05_Duality_and_Convex_Programming/                    ← planned
│   └── 06_Dual_Methods/                                      ← planned
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

Every notebook follows the architecture defined in `ARCH-RULES.md`:

| Cell | Type     | Content |
|------|----------|---------|
| 1    | Markdown | Section number and title |
| 2    | Markdown | Core equation(s) |
| 3    | Markdown | Theoretical explanation |
| 4    | Markdown | Numerical example with intermediate steps |
| 5    | Code     | Minimal implementation |
| 6    | Code     | Optional advanced analysis or visualization |
| 7    | Markdown | Optional extended theory or variants |
| 8    | Markdown | References + Previous / Next links |

Notebooks are sequentially linked — each one points to the previous and next in the series.

---

## Current Progress

| Section | Topic | Notebooks | Status |
|---------|-------|-----------|--------|
| 01 Foundations / 01 Linear Algebra | Prerequisites | 3 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Vectors | 9 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Matrices | 20 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Computational Linear Algebra | 6 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Geometrical Aspects | 17 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Linear Transformations | 14 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Theoretical Linear Algebra | 31 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Coordinate Transformations | 10 | ✅ Available |
| 01 Foundations / 01 Linear Algebra | Multilinear Algebra and Tensors | 13 | ✅ Available |
| 01 Foundations / 02–06 | Probability, Calculus, Variations, Differential Equations, Integral Transforms | 0 | 📋 Planned |
| 02 Dynamics | Causal/Acausal, Lagrange, Bond Graph, Port-Hamiltonian | 0 | 📋 Planned |
| 03 Optimization / 01 Unconstrained Optimization | Nonlinear programming, Chapter 1-style unconstrained methods | 30 | ✅ Available |
| 03 Optimization / 02 Convex Optimization | Boyd and Vandenberghe-style convex optimization sequence | 106 | ✅ Available |
| 03 Optimization / 03 Lagrange Multiplier Theory | Equality multipliers, KKT, Fritz John, constraint qualifications, duality | 17 | ✅ Available |
| 03 Optimization / 04–06 | Lagrange multiplier algorithms, duality and convex programming, dual methods | 0 | 📋 Planned |
| 04 Optimal Control | Controllability, estimation, LQR, LQG, trajectory optimization, MPC, robust control, hybrid systems, PDE-constrained optimal control | 0 | 📋 Planned |
| 05 Machine Learning | Core modules | 0 | 📋 Planned |
| 07 Reinforcement Learning | Core modules | 0 | 📋 Planned |
| A1 OOP | Object-oriented principles and design patterns | 13 | ✅ Available |
| A2 LLVM Compilers | Compiler foundations, LLVM IR, passes, optimization, MLIR | 16 | ✅ Available |
| A3 Performance and Systems | CPU performance, vectorization, parallelism, profiling | 9 | ✅ Available |
| A4 ML Infrastructure | ML compilers, framework integration, runtime topics, benchmarking | 12 | ✅ Available |
| A5 Interview Katas | OOP and machine-learning infrastructure katas | 28 | ✅ Available |

**Current Total: 354 notebooks**

---

## Missing Prerequisite Notes

The optimization notebooks now link to existing prerequisite notebooks where the repository already has coverage. A few foundational concepts still do not have dedicated prerequisite notebooks:

| Concept area | Needed in optimization notebooks | Likely foundation location |
|--------------|----------------------------------|----------------------------|
| Differential calculus for optimization | Gradients, Hessians, directional derivatives, Taylor approximations, implicit-function and mean-value theorem tools, sensitivity derivatives, primal functions, and second-order optimality conditions in `03_Optimization/01_Unconstrained_Optimization`, `03_Optimization/02_Convex_Optimization/09_Unconstrained_Minimization`, `10_Equality_Constrained_Minimization`, `11_Interior_Point_Methods`, and `03_Optimization/03_Lagrange_Multiplier_Theory` | `01_Foundations/03_Calculus` |
| Probability and statistics for convex estimation | Likelihoods, log-likelihoods, entropy, KL divergence, random variables, means, variances, covariance matrices, and hypothesis-testing terminology in `03_Optimization/02_Convex_Optimization/07_Statistical_Estimation` | `01_Foundations/02_Probability_and_Statistics` |
| Convex-analysis and variational-geometry bridge topics | Subgradients, supporting hyperplanes, separating hyperplanes, epigraph reasoning, dual cones, polar cones, tangent cones, Farkas' lemma, quasiregularity, quasinormality, constraint qualifications, and semi-infinite active-gradient cones across `03_Optimization/02_Convex_Optimization/02_Convex_Sets`, `03_Convex_Functions`, `05_Duality`, and `03_Optimization/03_Lagrange_Multiplier_Theory` | Either remain in `03_Optimization/02_Convex_Optimization` or become a future foundations-level convex-analysis bridge |

---

# Getting Started 


> [!IMPORTANT]
> **You don't need to install anything to explore this hub.**
> All notebooks render directly on GitHub — Just **leave a Star** and enjoy the ride! 🚀

> The setup guide below is **only** for those who want to **run the Python code**, **modify notebooks**, or **experiment locally**. And assumes **no prior developer setup** and walks through everything from scratch.


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
- Aazi, M. (2024). *Mathematics For Machine Learning*
- Rozycki, P. (2020). *Computational Mechanics Course Notes, École Centrale de Nantes*
