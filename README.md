# Optimal F1 Strategy Prediction  

## Project Description  

This project explores how **machine learning** can be used to predict **optimal Formula 1 race strategies** —  
including the **number of stints**, **stint lengths**, and **tire compounds** most likely to yield the best race results.  

The analysis and data preparation are built around the open-source [**Fast-F1**](https://github.com/theOehrly/Fast-F1) Python library,  
which allows access to publicly available Formula 1 timing and telemetry data.  

All datasets used in this project are **generated locally** from the Fast-F1 API — meaning the project **does not redistribute official Formula 1 data**,  
but instead **derives structured datasets** for research and educational purposes based on data accessible via Fast-F1.  

---

## Dataset  

**Source:** Self-generated using the [Fast-F1](https://github.com/theOehrly/Fast-F1) library (MIT License).  

### Data generation process  
1. Fetch timing and telemetry data via Fast-F1 API.  
2. Parse session data into structured pandas DataFrames.  
3. Extract stint lengths, compound usage, and pit stop information.  
4. Store the processed data in a reproducible local dataset for machine learning analysis.  

---

## License  

**Code:**  
This repository and its original code are distributed under the **MIT License**,  
as is the underlying [Fast-F1](https://github.com/theOehrly/Fast-F1) library.  

**Data:**  
The derived datasets are created using data accessed through Fast-F1,  
which itself is not affiliated with or endorsed by Formula One Licensing B.V.  
Users are responsible for ensuring compliance with Formula 1’s data use terms if redistributing or commercializing derived works.  

---

## How to run project? 

### 1. Clone repository 
```bash
git clone https://github.com/PJ-s28184/asi-project.git
cd asi-project
```

### 2.Create environment 
```bash
conda env create -f environment.yml
conda activate asi-ml
```

### 3. Run kedro 
```bash
kedro run
```
