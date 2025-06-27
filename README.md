# NCD_ISMIR_2025
This repository contains the official codebase for our paper: "[IDENTIFICATION AND CLUSTERING OF UNSEEN RAGAS IN INDIAN ART MUSIC]"
📄 [https://doi.org/10.48550/arXiv.2411.18611] [P. Singh, A. Gupta, A. Mishra and V. Arora, "IDENTIFICATION AND CLUSTERING OF UNSEEN RAGAS IN INDIAN ART MUSIC," in ISMIR, 2025]

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

This project addresses the challenge of Indian raga classification in realistic, open-world scenarios where unknown ragas may appear at test time. We propose and evaluate OOD detection and NCD clustering frameworks, benchmarked on the Saraga and PIM datasets. Our work is the first to explore these open-set paradigms for Indian classical music, providing robust baselines and reproducible code.

---

## Repository Structure

- `src/ood_main/` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; OOD detection code and scripts  
- `src/ncd/` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; NCD clustering code and scripts  
- `requirements.txt` &nbsp; Python dependencies  
- `README.md` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This documentation  

---

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/ParampreetSingh97/NCD_ISMIR_2025.git
cd <ParampreetSingh97/NCD_ISMIR_2025/>
pip install -r requirements.txt

## Requirements

This script requires [PIM_v1_XAI-repo](https://github.com/ParampreetSingh97/PIM_v1_XAI.git).

Clone it with:

```bash
git clone https://github.com/ParampreetSingh97/PIM_v1_XAI.git Pim_XAI
---
## Datasets

We use the publicly available Saraga and PIM datasets available at: .


---
## Citation

If you use this code or refer to our benchmarks, please cite: [P. Singh, A. Gupta, A. Mishra and V. Arora, "IDENTIFICATION AND CLUSTERING OF UNSEEN RAGAS IN INDIAN ART MUSIC," in ISMIR, 2025]

---
## Contact

For questions or contributions, please contact [params21@iitk.ac.in](mailto:params21@iitk.ac.in).

