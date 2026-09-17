import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATASET_FILE = "housing_dataset.csv"

# Helper function to format INR currency nicely
def format_inr(lakhs_val):
    if lakhs_val >= 100:
        crores = lakhs_val / 100
        return f"₹ {crores:.2f} Cr (₹ {lakhs_val:.2f} Lakhs)"
    else:
        return f"₹ {lakhs_val:.2f} Lakhs"

# ---------------------------------------------------------
# 1. DATASET SETUP & GENERATION ENGINE (INDIAN MARKET)
# ---------------------------------------------------------
def ensure_dataset_exists():
    """Generates a realistic synthetic Indian Housing Dataset if missing or outdated."""
    regenerate = False

    if os.path.exists(DATASET_FILE):
        try:
            df_check = pd.read_csv(DATASET_FILE)
            if 'price_lakhs' not in df_check.columns:
                print("[!] Outdated dataset schema detected. Regenerating with ₹ Lakhs...")
                regenerate = True
        except Exception:
            regenerate = True
    else:
        regenerate = True

    if not regenerate:
        return

    print("[+] Generating Indian housing market dataset...")
    np.random.seed(42)
    n_samples = 1000

    sqft = np.random.randint(500, 4000, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.randint(1, 5, size=n_samples)
    age = np.random.randint(0, 30, size=n_samples)
    distance_km = np.round(np.random.uniform(1.0, 25.0, size=n_samples), 1)

    # Synthetic price formula in Lakhs (₹)
    # Approx: 1000 sqft house ~ ₹ 60-90 Lakhs depending on location and age
    price_lakhs = (
        (sqft * 0.055) +
        (bedrooms * 5.5) +
        (bathrooms * 3.5) -
        (age * 0.4) -
        (distance_km * 1.2) +
        np.random.normal(15, 5, size=n_samples)
    )
    
    # Ensure minimum property price threshold (₹ 15 Lakhs)
    price_lakhs = np.round(np.maximum(price_lakhs, 15.0), 2)

    df = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'distance_km': distance_km,
        'price_lakhs': price_lakhs
    })
    df.to_csv(DATASET_FILE, index=False)
    print(f"[✓] Dataset created successfully as '{DATASET_FILE}'")

# ---------------------------------------------------------
# 2. MODEL TRAINING & EVALUATION ENGINE
# ---------------------------------------------------------
def train_and_evaluate():
    ensure_dataset_exists()
    df = pd.read_csv(DATASET_FILE)
    df.columns = df.columns.str.strip()  # Remove any accidental whitespace

    X = df.drop(columns=['price_lakhs'])
    y = df['price_lakhs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Machine Learning Models
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Evaluation Metrics
    lr_r2 = r2_score(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

    rf_r2 = r2_score(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    return {
        'scaler': scaler,
        'lr_model': lr_model,
        'rf_model': rf_model,
        'lr_r2': lr_r2,
        'lr_rmse': lr_rmse,
        'rf_r2': rf_r2,
        'rf_rmse': rf_rmse
    }

# ---------------------------------------------------------
# 3. INTERACTIVE PREDICTION ENGINE
# ---------------------------------------------------------
def predict_price(artifacts):
    print("\n" + "="*60)
    print("           INDIAN HOUSE PRICE PREDICTION INPUT             ")
    print("="*60)

    try:
        sqft = float(input("Enter Built-up Area in Sq Ft (e.g., 1200): "))
        bedrooms = int(input("Enter Number of Bedrooms / BHK (e.g., 2 or 3): "))
        bathrooms = int(input("Enter Number of Bathrooms (e.g., 2): "))
        age = float(input("Enter Age of Property in Years (e.g., 4): "))
        distance = float(input("Enter Distance to City Center / IT Hub in Km (e.g., 6.5): "))

        input_data = pd.DataFrame([[sqft, bedrooms, bathrooms, age, distance]],
                                  columns=['sqft', 'bedrooms', 'bathrooms', 'age', 'distance_km'])

        # Predict with Random Forest Model
        rf_price_lakhs = artifacts['rf_model'].predict(input_data)[0]

        # Predict with Linear Regression Model
        scaled_input = artifacts['scaler'].transform(input_data)
        lr_price_lakhs = artifacts['lr_model'].predict(scaled_input)[0]

        avg_price_lakhs = (rf_price_lakhs + lr_price_lakhs) / 2

        print("\n" + "-"*60)
        print("             ESTIMATED INDIAN PROPERTY VALUATION            ")
        print("-"*60)
        print(f"🏡 Random Forest Valuation  : {format_inr(rf_price_lakhs)}")
        print(f"📈 Linear Regression Valuation: {format_inr(lr_price_lakhs)}")
        print(f"📊 Average Estimated Price  : {format_inr(avg_price_lakhs)}")
        print("-" * 60 + "\n")

    except ValueError:
        print("\n[!] Invalid input! Please enter numeric values only.")

# ---------------------------------------------------------
# 4. MAIN MENU CONTROLLER
# ---------------------------------------------------------
def main():
    ensure_dataset_exists()
    print("\n[+] Training Machine Learning Regression Models...")
    artifacts = train_and_evaluate()
    print("[✓] Training Complete!")

    while True:
        print("="*60)
        print("   AI & ML: INDIAN HOUSE PRICE PREDICTION SYSTEM (CLI)     ")
        print("="*60)
        print(" 1. Predict Price for Property (in ₹ Rupees / Lakhs)")
        print(" 2. View Model Accuracy & Benchmark Metrics")
        print(" 3. View Dataset Summary")
        print(" 4. Exit Application")
        print("="*60)

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            predict_price(artifacts)

        elif choice == "2":
            print("\n" + "="*60)
            print("                MODEL BENCHMARK METRICS                   ")
            print("="*60)
            print(f"1. Linear Regression:")
            print(f"   • R² Score (Accuracy): {artifacts['lr_r2']*100:.2f}%")
            print(f"   • Root Mean Squared Error (RMSE): ₹ {artifacts['lr_rmse']:.2f} Lakhs")
            print(f"\n2. Random Forest Regressor:")
            print(f"   • R² Score (Accuracy): {artifacts['rf_r2']*100:.2f}%")
            print(f"   • Root Mean Squared Error (RMSE): ₹ {artifacts['rf_rmse']:.2f} Lakhs")
            print("="*60 + "\n")

        elif choice == "3":
            df = pd.read_csv(DATASET_FILE)
            print("\n" + "="*60)
            print("                 DATASET PREVIEW & SUMMARY                 ")
            print("="*60)
            print(f"Total Records: {len(df)} properties")
            print("\nFirst 5 Records:")
            print(df.head())
            print("\nSummary Statistics (Price in ₹ Lakhs):")
            print(df.describe().round(2))
            print("="*60 + "\n")

        elif choice == "4":
            print("\n[✓] Exiting application. Session terminated successfully.")
            sys.exit(0)
        else:
            print("\n[!] Invalid selection. Please choose an option between 1 and 4.")

if __name__ == "__main__":
    main()