# Flight Delay Analysis 

This project cleans, analyzes, and visualizes flight data using Python, Pandas, and PCA.

---

## Features
- Data cleaning and preprocessing
- Splitting `From_To` into separate `From` and `To` columns
- Creating new features (`TotalDelays`, `NumDelays`, `AverageDelay`)
- Statistical and visual analysis of delays
- Dimensionality reduction with PCA

---

## Installation
1. Clone this repository:
```bash
git clone https://github.com/KouroshBeheshtinejad/flight-delay-analysis-git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage
```bash
python analysis.py
```

---

## Outputs
The project will save the following plots in the `images/` folder:

- Histogram of Numeric Columns  
- Boxplot of Delay Columns  
- Scree Plot  
- Cumulative Explained Variance  
- Correlation Heatmap  
- PCA Scatter Plot  

---

## Project Structure
```bash
flight-delay-analysis/
│
├── datasetprojmabahes.csv
├── analysis.py
├── README.md
├── requirements.txt
└── images/
```

---

## Requirements
See `requirements.txt` for the full list

- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
