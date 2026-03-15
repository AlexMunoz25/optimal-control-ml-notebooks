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
├── 01_Foundations/
│   ├── 01_Linear_Algebra/                                    ← 104 notebooks
│   │   ├── 01_Prerequisites/                                 (3 notebooks)
│   │   ├── 02_Vector/                                        (9 notebooks)
│   │   ├── 03_Matrix/                                        (20 notebooks)
│   │   ├── 04_Computational_Linear_Algebra/                  (6 notebooks)
│   │   ├── 05_Geometrical_Aspects_of_Linear_Algebra/         (17 notebooks)
│   │   ├── 06_Linear_Transformations/                        (14 notebooks)
│   │   ├── 07_Theoretical_Linear_Algebra/                    (25 notebooks)
│   │   ├── 08_Coordinate_Transformations/                    (10 notebooks)
│   │   └── 09_Multilinear_Algebra_and_Tensors/               (planned)
│   ├── 02_Probability_and_Statistics/                        ← planned
│   ├── 03_Calculus/                                          ← planned
│   ├── 04_Calculus_of_Variations/                            ← planned
│   ├── 05_DIfferential_Equations/                            ← planned
│   └── 06_Integral_Transforms/                               ← planned
│
├── 02_Dynamics/
│   ├── 01_Causal_Acausal_Modeling/                           ← planned
│   ├── 02_Modeling_using_Lagrange/                           ← planned
│   ├── 03_Modeling_using_Bond_Graph/                         ← planned
│   └── 04_Modeling_using_Port_Hamiltonian/                   ← planned
│
├── 03_Optimization/
│   ├── 01_Linear_Programming_LP/                             ← planned
│   ├── 02_Convex_Quadratic_Programming_QP/                   ← planned
│   ├── 03_Convex_Quadratically_Constrained_Quadratic_Programming_QCQP/ ← planned
│   ├── 04_Second_Order_Cone_Programming_SOCP/                ← planned
│   ├── 05_Semidefinite_Programming_SDP/                      ← planned
│   ├── 06_Mixed_Integer_Programming_MIP/                     ← planned
│   ├── 07_Global_Nonconvex_Optimization/                     ← planned
│   ├── 08_Robust_and_Stochastic_Optimization/                ← planned
│   └── 09_PDE_Constrained_Optimization/                      ← planned
│
├── 04_Optimal_Control/
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
│   ├── 11_PDE_Constrained_Optimal_Control/                   ← planned
│   └── 13_Reinforcement_Learning_and_Approx_DP/              ← planned
│
├── 05_Machine_Learning/                                      ← planned
├── 07_Reinforcement_Learning/                                ← planned
│
├── A1_OOP/
│   ├── 01_Object_Oriented_Principle/
│   └── 02_Object_Oriented_Design_Patterns/
├── A2_LLVM_Compilers/
│   ├── 01_Compiler_Foundations/
│   ├── 02_LLVM_Intermediate_Representation/
│   ├── 03_Pass_Infrastructure/
│   ├── 04_Optimization_Topics/
│   └── 05_MLIR/
├── A3_Performance_and_Systems/
│   ├── 01_Central_Processing_Unit_Performance/
│   ├── 02_Vectorization/
│   ├── 03_Parallelism/
│   └── 04_Profiling/
├── A4_ML_Infrastructure/
│   ├── 01_ML_Compiler_Ecosystem/
│   ├── 02_Framework_Integration/
│   ├── 03_Runtime_Topics/
│   └── 04_Benchmarking_and_Regressions/
│
├── Figures/                                                  ← reusable diagrams and plots
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
| 4    | Code     | Minimal implementation |
| 5    | Code     | Optional analysis or visualization |
| 6    | Markdown | Optional extended theory or variants |
| 7    | Markdown | References + Previous / Next links |

Notebooks are sequentially linked — each one points to the previous and next in the series.

---

## Current Progress

| Section | Topic | Notebooks | Status |
|---------|-------|-----------|--------|
| 01 Foundations / 01 Linear Algebra | Prerequisites | 3 | ✅ |
| 01 Foundations / 01 Linear Algebra | Vectors | 9 | ✅ |
| 01 Foundations / 01 Linear Algebra | Matrices | 20 | ✅ |
| 01 Foundations / 01 Linear Algebra | Computational Linear Algebra | 6 | ✅ |
| 01 Foundations / 01 Linear Algebra | Geometrical Aspects | 17 | ✅ |
| 01 Foundations / 01 Linear Algebra | Linear Transformations | 14 | ✅ |
| 01 Foundations / 01 Linear Algebra | Theoretical Linear Algebra | 25 | ✅ |
| 01 Foundations / 01 Linear Algebra | Coordinate Transformations | 10 | ✅ |
| 01 Foundations / 01 Linear Algebra | Multilinear Algebra and Tensors | — | 📋 Planned |
| 01 Foundations / 02–06 | Probability, Calculus, Variations, Differential Equations, Integral Transforms | — | 📋 Planned |
| 02 Dynamics | Causal/Acausal, Lagrange, Bond Graph, Port-Hamiltonian | — | 📋 Planned |
| 03 Optimization | LP through PDE-Constrained Optimization | — | 📋 Planned |
| 04 Optimal Control | Controllability, Estimation, LQR, MPC, Robust Control, RL | — | 📋 Planned |
| 05 Machine Learning | Core modules | — | 📋 Planned |
| 07 Reinforcement Learning | Core modules | — | 📋 Planned |
| Appendices | OOP, LLVM, Performance, ML Infrastructure | In progress | 🚧 |

**Current Total: 154 notebooks**

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
