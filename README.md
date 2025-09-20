# NNS

This is a local copy of the [SCNS](https://github.com/Annjnk/scns) project, configured to run on Apple Silicon (M1/M2) Macs.

---

## 🧩 Environment Setup (macOS M1/M2)

This project uses:

- Python 3.9  
- Mayavi 4.8.1  
- VTK 9.1.0  
- PyQt5 (GUI backend)

### 1. Install Miniconda (if not installed)
Download from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create the Environment
Make sure you're in the project folder, then run:

```bash

conda env create -f environment.yml
conda activate scns_env
