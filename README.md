# 🏡 Indian House Price Prediction Engine (Supervised ML)

**An End-to-End Regression Pipeline & Interactive Terminal CLI Application**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat-square&logo=scikit-learn)
![Interface](https://img.shields.io/badge/Interface-CLI%20Terminal-green?style=flat-square)
![Currency](https://img.shields.io/badge/Currency-%E2%82%B9%20INR%20(Lakhs%2FCrores)-brightgreen?style=flat-square)
![Evaluation](https://img.shields.io/badge/Evaluation-VITyarthi--AIML-purple?style=flat-square)

---

# 👤 Author & Project Metadata

| Detail | Information |
| :--- | :--- |
| **Project Name** | Indian House Price Prediction Engine |
| **Lead Developer** | Sahil Ahirwar |
| **Registration Number** | 25MIM10032 |
| **Course Code / Name** | Fundamentals in AI & Machine Learning |
| **Institution** | VIT - VITyarthi Programme |
| **Project Category** | Supervised Machine Learning (Regression) |

---

# 📖 Project Overview

Accurate real estate valuation is a core application of predictive machine learning. This project implements an end-to-end Machine Learning regression pipeline built in Python using `scikit-learn`, designed specifically for the Indian housing market.

The system trains, benchmarks, and evaluates two regression models—**Linear Regression** and **Random Forest Regressor**—to predict housing valuations in **Indian Rupees (₹ Lakhs & ₹ Crores)** based on property parameters such as built-up area (Sq Ft), BHK configuration, bathrooms, property age, and proximity to city centers or major IT hubs.

---

# ⚙️ Key System Capabilities

* **Automated Data Bootstrapping:** Automatically checks and generates a realistic 1,000-record Indian housing dataset (`housing_dataset.csv`) upon first launch.
* **Dual Model Benchmarking:** Trains both **Linear Regression** (with `StandardScaler`) and **Random Forest Regressor** to evaluate accuracy using **$R^2$ Score** and **Root Mean Squared Error (RMSE)**.
* **Smart Currency Formatting:** Dynamic valuation output that formats results in **Lakhs (₹ L)** or converts to **Crores (₹ Cr)** if property value exceeds ₹ 100 Lakhs.
* **100% Zero-GUI Terminal Execution:** Designed for headless automated grader pipelines.

---

# 🛠️ Tech Stack & Dependencies

- **Programming Language:** Python 3.8+
- **Machine Learning Engine:** `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **User Interface:** Interactive Command-Line Interface (CLI)

---

# 🚀 Setup & Execution Instructions (macOS / Linux / Windows)

### 1. Clone the Repository
```bash
git clone [https://github.com/sahil25mim10032/AIML-House-Price-Predictor.git](https://github.com/sahil25mim10032/AIML-House-Price-Predictor.git)
cd AIML-House-Price-Predictor
```

### 2. Set Up Virtual Environment & Install Dependencies (Mac)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python3 main.py
```

---

# 📊 Interactive CLI Menu Structure

1. **Predict Price for Property (in ₹ Rupees / Lakhs):** Interactive prompt where users input property specifications (Sq Ft, BHK, Bathrooms, Age, Distance) to receive live valuations.
2. **View Model Accuracy & Benchmark Metrics:** Displays $R^2$ accuracy scores and Root Mean Squared Error (RMSE) in ₹ Lakhs for both models.
3. **View Dataset Summary:** Previews property dataset statistics, feature distributions, and record head.
4. **Exit Application:** Cleanly terminates the terminal session.

<!-- contributor-refresh -->
