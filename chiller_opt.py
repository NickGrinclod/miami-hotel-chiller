from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Folder containing our CSV datasets
data_folder = Path("data")

# Find all CSV files
csv_files = sorted(data_folder.glob("*.csv"))

print( csv_files)

print("Number of CSV files:", len(csv_files))

for file in csv_files:
    print(file.name)

import pandas as pd

# Empty list to store the DataFrames
dataframes = []

for file in csv_files:

    # Extract setpoint from filename
    # Example: baseline_12C_results.csv -> 12
    setpoint = float(file.stem.split("_")[2].replace("C", ""))

     # Read CSV file
    df = pd.read_csv(file)
    
        # Add setpoint as a new feature
    df["setpoint_C"] = setpoint
    
        # Add DataFrame to list
    dataframes.append(df)
    
    # Combine all five datasets vertically
combined_df = pd.concat(dataframes, ignore_index=True)
    
print("\nCombined dataset shape:")
print(combined_df.shape)
    
print("\nFirst five rows:")
print(combined_df.head())
    
print("\nColumn names:")
print(combined_df.columns.tolist())

print("\nMissing values by column:")
print(combined_df.isna().sum())

# Load the three CSV files
df_67 = pd.read_csv(data_folder / "Miami_baseline_6.7C.csv")
df_7 = pd.read_csv(data_folder / "Miami_Hotel_7C.csv")
df_8 = pd.read_csv(data_folder / "Miami_Hotel_8C.csv")

# Compare all Date/Time values
print("6.7°C vs 8°C:",
      df_67["Date/Time"].equals(df_8["Date/Time"]))

print("7°C vs 8°C:",
      df_7["Date/Time"].equals(df_8["Date/Time"]))

# Find timestamps that differ between baseline and 8C
different = df_67["Date/Time"] != df_8["Date/Time"]

print("\nNumber of different timestamps:", different.sum())

print("\nFirst differences:")
print(pd.DataFrame({
    "Baseline": df_67.loc[different, "Date/Time"],
    "8C": df_8.loc[different, "Date/Time"]
}).head(10))

df_9 = pd.read_csv(data_folder / "Miami_Hotel_9C.csv")
df_10 = pd.read_csv(data_folder / "Miami_Hotel_10C.csv")

print("9°C vs 8°C:",
      df_9["Date/Time"].equals(df_8["Date/Time"]))

print("10°C vs 8°C:",
      df_10["Date/Time"].equals(df_8["Date/Time"]))

weather_cols = [
    "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)",
    "Environment:Site Outdoor Air Relative Humidity [%](Hourly)"
]

for name, df in [
    ("6.7C", df_67),
    ("7C", df_7),
    ("9C", df_9),
    ("10C", df_10)
]:
    print(name, "weather matches 8C:",
          df[weather_cols].equals(df_8[weather_cols]))

# Check missing solar radiation values in the 8C dataset

solar_cols = [
    "Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)",
    "Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)"
]

print("\nMissing solar values in 8C dataset:")
print(df_8[solar_cols].isna().sum())

# Find the column numbers for solar radiation
for col in solar_cols:
    print("Column number:", combined_df.columns.get_loc(col) + 1)
    print("Column name:", col)


# Find solar radiation columns in the original 8C CSV
for col in solar_cols:
    print("8C column number:", df_8.columns.get_loc(col) + 1)
    print("Column name:", col)


# Fill missing solar radiation values using the 8C dataset
solar_reference = df_8.set_index("Date/Time")[solar_cols]

for col in solar_cols:
    combined_df[col] = combined_df[col].fillna(
        combined_df["Date/Time"].map(solar_reference[col])
    )

# Check that no solar values are missing
print("\nMissing solar values after filling:")
print(combined_df[solar_cols].isna().sum())



energy_col = "COOLSYS1 CHILLER1:Chiller Electricity Energy [J](Hourly)"

combined_df["Chiller_kWh"] = combined_df[energy_col] / 3_600_000

print(combined_df[["setpoint_C", "Chiller_kWh"]].head())
# Save the combined dataset
output_file = "Miami_Hotel_Combined.csv"

combined_df.to_csv(output_file, index=False)

print("\nSaved combined dataset:", output_file)
print("Dataset shape:", combined_df.shape)

# Find columns related to cooling load
print("\nCooling-related columns:")

for col in combined_df.columns:
    if any(word in col.lower() for word in ["cooling", "load", "chiller"]):
        print(col)


# Calculate average indoor conditions across hotel zones

#avg_indoor_temp = combined_df["room_temp_c"].mean()
#avg_indoor_humidity = combined_df["room_humidity_perc"].mean()

#print(f"Average indoor temperature: {avg_indoor_temp:.2f} °C")
#print(f"Average indoor humidity: {avg_indoor_humidity:.2f} %")


# Find indoor temperature and humidity columns

for col in combined_df.columns:
    if any(word in col.lower() for word in ["temp", "humid", "zone"]):
        print(col)


# Find all hotel zone temperature columns
temp_cols = [
    col for col in combined_df.columns
    if "Zone Mean Air Temperature" in col
]

# Find all hotel zone humidity columns
humidity_cols = [
    col for col in combined_df.columns
    if "Zone Air Relative Humidity" in col
]

# Calculate hourly averages across all zones
combined_df["avg_indoor_temp"] = (
    combined_df[temp_cols].mean(axis=1)
)

combined_df["avg_indoor_humidity"] = (
    combined_df[humidity_cols].mean(axis=1)
)

# Display results
print("Temperature zones:", len(temp_cols))
print("Humidity zones:", len(humidity_cols))

print(
    combined_df[
        ["avg_indoor_temp", "avg_indoor_humidity"]
    ].head()
)


# Check hourly sequence and duplicate timestamps

# Step 1: choose the timestamp column
time_col = "Date/Time"

# Step 2: keep original text version
combined_df["time_str_original"] = combined_df[time_col].astype(str).str.strip()

# Step 3: create cleaned timestamp strings
time_str = combined_df["time_str_original"].copy()

# Find rows that contain 24:00:00
is_24hour = time_str.str.contains("24:00:00", na=False)

# Replace 24:00:00 with 00:00:00 temporarily
time_str_fixed = time_str.str.replace("24:00:00", "00:00:00", regex=False)

# Step 4: convert to datetime
# Using a dummy year because the dataset appears to contain only month/day time
combined_df["timestamp"] = pd.to_datetime(
    "2000/" + time_str_fixed,
    format="%Y/%m/%d %H:%M:%S",
    errors="coerce"
)

# Step 5: for original 24:00:00 rows, add one day
combined_df.loc[is_24hour, "timestamp"] = (
    combined_df.loc[is_24hour, "timestamp"] + pd.Timedelta(days=1)
)

# Step 6: check if any timestamps failed to parse
bad_timestamps = combined_df["timestamp"].isna().sum()
print("Unparsed timestamps:", bad_timestamps)

# Step 7: sort by cleaned timestamp
combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

# Step 8: check duplicates
duplicate_count = combined_df["timestamp"].duplicated().sum()
print("Duplicate timestamps:", duplicate_count)

# Step 9: check time gaps
time_diff = combined_df["timestamp"].diff()

irregular_mask = time_diff.notna() & (time_diff != pd.Timedelta(hours=1))
irregular_count = irregular_mask.sum()

print("Irregular intervals:", irregular_count)

# Step 10: show some problematic rows if they exist
if irregular_count > 0:
    print("\nFirst 10 irregular timestamps:")
    print(
        combined_df.loc[irregular_mask, ["time_str_original", "timestamp"]].head(10)
    )
else:
    print("All intervals are exactly 1 hour.")

# Optional: show first few cleaned timestamps
print("\nFirst 5 cleaned timestamps:")
print(combined_df[["time_str_original", "timestamp"]].head())


# ==========================================
# FINAL DATA VALIDATION BEFORE ML TRAINING
# ==========================================

print("\n=== FINAL DATA VALIDATION ===")

# Step 1: Find columns related to setpoint and energy
for col in combined_df.columns:
    if any(word in col.lower() for word in [
        "setpoint", "scenario", "energy", "electricity",
        "chiller", "cooling"
    ]):
        print(col)

# Step 2: Define our five temperature setpoints
expected_setpoints = [6.7, 7, 8, 9, 10]

print("\nExpected setpoints:", expected_setpoints)

# Step 3: Check timestamp conversion
print("\nTotal records:", len(combined_df))

print(
    "Invalid timestamps:",
    combined_df["timestamp"].isna().sum()
)

# Step 4: Count records for each timestamp
hourly_counts = combined_df.groupby("timestamp").size()

print("\nRecords per timestamp:")
print(hourly_counts.value_counts().sort_index())

# Step 5: Check missing values in every column
missing_values = combined_df.isna().sum()

print("\nColumns containing missing values:")
print(missing_values[missing_values > 0])


# ==========================================
# FIRST ML MODEL: PREDICT CHILLER ENERGY
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n=== TRAIN FIRST MODEL ===")

# Step 1: Create time-based features
combined_df["hour"] = combined_df["timestamp"].dt.hour
combined_df["month"] = combined_df["timestamp"].dt.month

# Step 2: Define feature columns
feature_cols = [
    "setpoint_C",
    "avg_indoor_temp",
    "avg_indoor_humidity",
    "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)",
    "Environment:Site Outdoor Air Relative Humidity [%](Hourly)",
    "hour",
    "month",
    "Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)",
    "Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)"

]

target_col = "Chiller_kWh"

# Step 3: Prepare X and y
X = combined_df[feature_cols]
y = combined_df[target_col]

print("\nFeature columns:")
print(feature_cols)

print("\nTarget column:")
print(target_col)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Step 4: Sort by time for chronological split
combined_df_sorted = combined_df.sort_values("timestamp").reset_index(drop=True)

X = combined_df_sorted[feature_cols]
y = combined_df_sorted[target_col]

# Step 5: Chronological train/test split (80% / 20%)
split_index = int(len(combined_df_sorted) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL RESULTS ===")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Step 9: Show coefficients
print("\n=== FEATURE COEFFICIENTS ===")
for feature, coef in zip(feature_cols, model.coef_):
    print(f"{feature}: {coef:.6f}")

print(f"\nIntercept: {model.intercept_:.6f}")

# Step 10: Show first 10 predictions vs actual
results_df = combined_df_sorted.iloc[split_index:].copy()
results_df["actual_kwh"] = y_test.values
results_df["predicted_kwh"] = y_pred

print("\n=== FIRST 10 PREDICTIONS ===")
print(results_df[["timestamp", "setpoint_C", "actual_kwh", "predicted_kwh"]].head(10))


# Check prediction bias

errors = y_test - y_pred

print("\n=== PREDICTION BIAS ===")
print("Average signed error:", errors.mean())
print("Maximum absolute error:", abs(errors).max())

print("\nAverage actual energy:", y_test.mean())
print("Average predicted energy:", y_pred.mean())



# Compare training and testing performance

train_pred = model.predict(X_train)

print("\n=== TRAINING PERFORMANCE ===")
print("Train MAE:", mean_absolute_error(y_train, train_pred))
print("Train R2:", r2_score(y_train, train_pred))

print("\n=== TESTING PERFORMANCE ===")
print("Test MAE:", mean_absolute_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))



# ==========================================
# TEST CHILLER ENERGY AT DIFFERENT SETPOINTS
# ==========================================

print("\n=== SETPOINT OPTIMIZATION TEST ===")

# Step 1: Define five chilled-water setpoints
setpoints = [6.7, 7, 8, 9, 10]

# Step 2: Define constant operating conditions
test_conditions = {
    "avg_indoor_temp": 23.0,
    "avg_indoor_humidity": 50.0,
    "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": 30.0,
    "Environment:Site Outdoor Air Relative Humidity [%](Hourly)": 60.0,
    "hour": 14,
    "month": 7,
    "Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)": 600.0,
    "Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)": 100.0

}

# Step 3: Create input records
test_rows = []

for sp in setpoints:
    row = test_conditions.copy()
    row["setpoint_C"] = sp
    test_rows.append(row)

# Step 4: Create a DataFrame
test_df = pd.DataFrame(test_rows)

# Keep columns in the same order used for training
test_df = test_df[feature_cols]

# Step 5: Predict chiller energy consumption
test_df["predicted_kWh"] = model.predict(test_df[feature_cols])

# Step 6: Calculate savings relative to 6.7°C baseline
baseline_kWh = test_df.loc[
    test_df["setpoint_C"] == 6.7,
    "predicted_kWh"
].iloc[0]

test_df["savings_kWh"] = (
    baseline_kWh - test_df["predicted_kWh"]
)

test_df["savings_percent"] = (
    test_df["savings_kWh"] / baseline_kWh * 100
)

# Step 7: Display results
print(
    test_df[
        ["setpoint_C", "predicted_kWh",
         "savings_kWh", "savings_percent"]
    ].to_string(index=False)
)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# NEURAL NETWORK MODEL
# ==========================================

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

# Train the neural network
nn_model.fit(X_train_scaled, y_train)

# Predict chiller energy using test data
nn_pred = nn_model.predict(X_test_scaled)

# Evaluate neural network performance
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
nn_r2 = r2_score(y_test, nn_pred)

print("\n=== NEURAL NETWORK RESULTS ===")
print("MAE:", nn_mae)
print("RMSE:", nn_rmse)
print("R2:", nn_r2)

# ==========================================
# CHECK NEURAL NETWORK OVERFITTING
# ==========================================

# Predict using training data
nn_train_pred = nn_model.predict(X_train_scaled)

# Calculate training metrics
nn_train_mae = mean_absolute_error(y_train, nn_train_pred)
nn_train_r2 = r2_score(y_train, nn_train_pred)

# Compare training and testing performance
print("\n=== NEURAL NETWORK: TRAIN VS TEST ===")

print("Training MAE:", nn_train_mae)
print("Testing MAE:", nn_mae)

print("Training R2:", nn_train_r2)
print("Testing R2:", nn_r2)

import matplotlib.pyplot as plt

# ==========================================
# ACTUAL VS PREDICTED CHILLER ENERGY
# ==========================================

plt.figure(figsize=(10, 6))

# Linear regression predictions
plt.scatter(
    y_test,
    y_pred,
    alpha=0.3,
    label="Linear Regression"
)

# Neural network predictions
plt.scatter(
    y_test,
    nn_pred,
    alpha=0.3,
    label="Neural Network"
)

# Perfect prediction reference line
min_energy = min(y_test.min(), y_pred.min(), nn_pred.min())
max_energy = max(y_test.max(), y_pred.max(), nn_pred.max())

plt.plot(
    [min_energy, max_energy],
    [min_energy, max_energy],
    "k--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Chiller Energy (kWh)")
plt.ylabel("Predicted Chiller Energy (kWh)")
plt.title("Miami Hotel — Actual vs Predicted Chiller Energy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("chiller_model_comparison.png", dpi=300)
plt.show()

# ==========================================
# CHILLER ENERGY VS TIME
# ACTUAL VS PREDICTED
# ==========================================

import matplotlib.pyplot as plt

# Get timestamps for the testing period
test_timestamps = combined_df_sorted["timestamp"].iloc[split_index:]

# Select the first 168 test records (example: 7 days
# if the data contains 24 hourly records per day)
n = 168

# Create graph
plt.figure(figsize=(14, 6))

# Actual chiller energy
plt.plot(
    test_timestamps.iloc[:n],
    y_test.iloc[:n],
    label="Actual Energy",
    color="black"
)

# Linear regression predictions
plt.plot(
    test_timestamps.iloc[:n],
    y_pred[:n],
    label="Linear Regression",
    color="blue"
)

# Neural network predictions
plt.plot(
    test_timestamps.iloc[:n],
    nn_pred[:n],
    label="Neural Network",
    color="orange"
)

plt.xlabel("Date and Time")
plt.ylabel("Chiller Energy (kWh)")
plt.title("Miami Hotel - Actual vs Predicted Chiller Energy Over Time")

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("chiller_energy_vs_time.png", dpi=300)
plt.show()
# ==========================================
# DAILY CHILLER ENERGY MODEL COMPARISON
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Create a DataFrame containing testing records
plot_df = combined_df_sorted.iloc[split_index:].copy()

# Add actual and predicted energy
plot_df["actual_kwh"] = y_test.to_numpy()
plot_df["linear_pred"] = np.asarray(y_pred).ravel()
plot_df["neural_pred"] = np.asarray(nn_pred).ravel()

# Extract the calendar date
plot_df["date"] = plot_df["timestamp"].dt.normalize()

# Calculate daily averages across all available setpoints
daily_df = (
    plot_df.groupby("date")[
        ["actual_kwh", "linear_pred", "neural_pred"]
    ]
    .mean()
    .reset_index()
)

# ==========================================
# GRAPH 1: ENTIRE TESTING PERIOD
# ==========================================

plt.figure(figsize=(16, 7))

plt.plot(
    daily_df["date"],
    daily_df["actual_kwh"],
    label="Actual Energy",
    color="black"
)

plt.plot(
    daily_df["date"],
    daily_df["linear_pred"],
    label="Linear Regression",
    color="blue"
)

plt.plot(
    daily_df["date"],
    daily_df["neural_pred"],
    label="Neural Network",
    color="orange"
)

plt.xlabel("Date")
plt.ylabel("Average Hourly Chiller Energy (kWh)")

plt.title(
    "Miami Hotel - Daily Average Chiller Energy"
)

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("chiller_daily_comparison.png", dpi=300)
plt.show()
# ==========================================
# SELECT TWO COMPARISON DAYS
# ==========================================

# Original October comparison day
october_day = pd.Timestamp("2000-10-20")

# Outdoor temperature column
outdoor_col = (
    "Environment:Site Outdoor Air Drybulb "
    "Temperature [C](Hourly)"
)

# Find the coldest December day if available
december_df = plot_df[
    plot_df["date"].dt.month == 12
]

if not december_df.empty:
    candidate_df = december_df
else:
    candidate_df = plot_df

# Calculate daily average outdoor temperatures
daily_outdoor = candidate_df.groupby("date")[outdoor_col].mean()

cold_day = daily_outdoor.idxmin()

print("Selected colder day:", cold_day)

# ==========================================
# FUNCTION TO PLOT AND EVALUATE ONE DAY
# ==========================================

def compare_day(selected_date):

    day_df = plot_df[
        plot_df["date"] == selected_date
    ].copy()

    if day_df.empty:
        print("No testing records for:", selected_date)
        return

    # Sort records by timestamp
    day_df = day_df.sort_values(
        ["timestamp", "setpoint_C"]
    )

    # Create hourly graph
    plt.figure(figsize=(14, 6))

    plt.plot(
        day_df["timestamp"],
        day_df["actual_kwh"],
        label="Actual Energy",
        color="black"
    )

    plt.plot(
        day_df["timestamp"],
        day_df["linear_pred"],
        label="Linear Regression",
        color="blue"
    )

    plt.plot(
        day_df["timestamp"],
        day_df["neural_pred"],
        label="Neural Network",
        color="orange"
    )

    plt.xlabel("Date and Time")
    plt.ylabel("Chiller Energy (kWh)")

    plt.title(
        f"Miami Hotel - {selected_date.date()}"
    )

    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"chiller_{selected_date.date()}.png",
        dpi=300
    )

    plt.show()

    # ======================================
    # CALCULATE MODEL PERFORMANCE
    # ======================================

    actual = day_df["actual_kwh"]

    for name, column in [
        ("Linear Regression", "linear_pred"),
        ("Neural Network", "neural_pred")
    ]:

        predicted = day_df[column]

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = r2_score(actual, predicted)

        print(f"\n{name} - {selected_date.date()}")
        print(f"MAE:  {mae:.4f} kWh")
        print(f"RMSE: {rmse:.4f} kWh")
        print(f"R2:   {r2:.4f}")


# Compare both days
compare_day(october_day)
compare_day(cold_day)

# ==========================================
# MONTHLY MODEL PERFORMANCE COMPARISON
# OCTOBER, NOVEMBER, DECEMBER
# ==========================================

# Create a DataFrame using testing records
monthly_df = combined_df_sorted.iloc[split_index:].copy()

# Add actual and predicted energy values
monthly_df["actual_kwh"] = y_test.to_numpy()
monthly_df["linear_pred"] = y_pred
monthly_df["neural_pred"] = nn_pred

# Extract month from timestamp
monthly_df["month"] = monthly_df["timestamp"].dt.month

# Define months to evaluate
months = {
    10: "October",
    11: "November",
    12: "December"
}

# Loop through each month
for month_number, month_name in months.items():

    # Select only records belonging to this month
    month_df = monthly_df[
        monthly_df["month"] == month_number
    ]

    if month_df.empty:
        print(f"No testing data for {month_name}")
        continue

    print(f"\n========== {month_name.upper()} ==========")
    print("Testing records:", len(month_df))

    # Actual energy consumption
    actual = month_df["actual_kwh"]

    # Evaluate both models
    for model_name, column_name in [
        ("Linear Regression", "linear_pred"),
        ("Neural Network", "neural_pred")
    ]:

        predicted = month_df[column_name]

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = r2_score(actual, predicted)

        print(f"\n{model_name}")
        print(f"MAE:  {mae:.4f} kWh")
        print(f"RMSE: {rmse:.4f} kWh")
        print(f"R²:   {r2:.4f}")


        # ==========================================
# ANALYZE WHETHER MODEL PERFORMANCE DIFFERENCE
# IS RELATED TO ENERGY CONSUMPTION
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Build a results DataFrame from the TEST set
analysis_df = pd.DataFrame({
    "Actual_kWh": np.asarray(y_test).ravel(),
    "LR_Pred_kWh": np.asarray(y_pred).ravel(),
    "NN_Pred_kWh": np.asarray(nn_pred).ravel()
}).reset_index(drop=True)

# Add timestamps from the TEST portion
analysis_df["timestamp"] = (
    combined_df_sorted.iloc[split_index:]["timestamp"]
    .reset_index(drop=True)
)

# Add month
analysis_df["Month"] = analysis_df["timestamp"].dt.month_name()

# Absolute errors
analysis_df["LR_Abs_Error"] = abs(
    analysis_df["LR_Pred_kWh"] - analysis_df["Actual_kWh"]
)

analysis_df["NN_Abs_Error"] = abs(
    analysis_df["NN_Pred_kWh"] - analysis_df["Actual_kWh"]
)

# Signed errors (optional, useful if you also want over/under prediction)
analysis_df["LR_Error"] = (
    analysis_df["LR_Pred_kWh"] - analysis_df["Actual_kWh"]
)

analysis_df["NN_Error"] = (
    analysis_df["NN_Pred_kWh"] - analysis_df["Actual_kWh"]
)

# ==========================================
# 1. PERFORMANCE BY ENERGY CONSUMPTION RANGE
# ==========================================

bins = [-np.inf, 25, 50, 75, 100, 125, np.inf]
labels = ["0-25", "25-50", "50-75", "75-100", "100-125", "125+"]

analysis_df["Load_Range"] = pd.cut(
    analysis_df["Actual_kWh"],
    bins=bins,
    labels=labels
)

load_results = analysis_df.groupby(
    "Load_Range",
    observed=True
).agg(
    LR_MAE=("LR_Abs_Error", "mean"),
    NN_MAE=("NN_Abs_Error", "mean"),
    Samples=("Actual_kWh", "size")
)

load_results["Best_Model"] = np.where(
    load_results["LR_MAE"] < load_results["NN_MAE"],
    "Linear Regression",
    "Neural Network"
)

print("\n=== PERFORMANCE BY CHILLER LOAD RANGE ===")
print(load_results.round(2))

# ==========================================
# 2. PERFORMANCE BY MONTH AND LOAD RANGE
# ==========================================

monthly_load_results = analysis_df.groupby(
    ["Month", "Load_Range"],
    observed=True
).agg(
    LR_MAE=("LR_Abs_Error", "mean"),
    NN_MAE=("NN_Abs_Error", "mean"),
    Samples=("Actual_kWh", "size")
)

monthly_load_results["Best_Model"] = np.where(
    monthly_load_results["LR_MAE"] < monthly_load_results["NN_MAE"],
    "Linear Regression",
    "Neural Network"
)

print("\n=== PERFORMANCE BY MONTH AND LOAD RANGE ===")
print(monthly_load_results.round(2))

# ==========================================
# 3. BAR CHART: MODEL ACCURACY BY LOAD RANGE
# ==========================================

ax = load_results[["LR_MAE", "NN_MAE"]].plot(
    kind="bar",
    figsize=(11, 6),
    color=["steelblue", "seagreen"],
    width=0.8
)

plt.xlabel("Actual Chiller Energy Range (kWh)")
plt.ylabel("Mean Absolute Error (kWh)")
plt.title("Model Accuracy by Chiller Load Range")
plt.xticks(rotation=0)
plt.legend(["Linear Regression", "Neural Network"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_accuracy_by_load_range.png", dpi=300)
plt.show()

# ==========================================
# 4. OPTIONAL: BAR CHART FOR A SPECIFIC MONTH
#    Example: October
# ==========================================

month_to_plot = "October"

month_df = monthly_load_results.loc[month_to_plot]

ax = month_df[["LR_MAE", "NN_MAE"]].plot(
    kind="bar",
    figsize=(11, 6),
    color=["steelblue", "seagreen"],
    width=0.8
)

plt.xlabel("Actual Chiller Energy Range (kWh)")
plt.ylabel("Mean Absolute Error (kWh)")
plt.title(f"Model Accuracy by Load Range - {month_to_plot}")
plt.xticks(rotation=0)
plt.legend(["Linear Regression", "Neural Network"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"model_accuracy_by_load_range_{month_to_plot}.png", dpi=300)
plt.show()

# ==========================================
# TRAINING VS TESTING DATA DISTRIBUTION
# BY CHILLER ENERGY CONSUMPTION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define energy consumption ranges
bins = [-np.inf, 25, 50, 75, 100, 125, np.inf]

labels = [
    "Below 25",
    "25-50",
    "50-75",
    "75-100",
    "100-125",
    "Above 125"
]

# Step 2: Create DataFrames for training and testing
train_df = pd.DataFrame({
    "Energy": np.asarray(y_train).ravel()
})

test_df = pd.DataFrame({
    "Energy": np.asarray(y_test).ravel()
})

# Step 3: Assign every record to an energy range
train_df["Load_Range"] = pd.cut(
    train_df["Energy"],
    bins=bins,
    labels=labels
)

test_df["Load_Range"] = pd.cut(
    test_df["Energy"],
    bins=bins,
    labels=labels
)

# Step 4: Count records in each energy range
train_counts = train_df["Load_Range"].value_counts().sort_index()
test_counts = test_df["Load_Range"].value_counts().sort_index()

# Step 5: Calculate percentages
train_percent = train_counts / len(train_df) * 100
test_percent = test_counts / len(test_df) * 100

# Step 6: Combine results into one table
distribution = pd.DataFrame({
    "Train_Count": train_counts,
    "Train_Percent": train_percent,
    "Test_Count": test_counts,
    "Test_Percent": test_percent
})

print("\n=== CHILLER ENERGY DATA DISTRIBUTION ===")
print(distribution.round(2).to_string())

# Step 7: Visualize the distribution
ax = distribution[["Train_Percent", "Test_Percent"]].plot(
    kind="bar",
    figsize=(11, 6),
    color=["steelblue", "seagreen"],
    width=0.8
)

plt.xlabel("Actual Chiller Energy Consumption (kWh)")
plt.ylabel("Percentage of Records (%)")
plt.title("Miami Hotel - Training vs Testing Load Distribution")
plt.xticks(rotation=0)
plt.legend(["Training Data", "Testing Data"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("chiller_train_test_distribution.png", dpi=300)
plt.show()

# ==========================================
# EXPERIMENT: LOW-LOAD NEURAL NETWORK
# ==========================================

# Define low-energy threshold
low_load_limit = 75

# Identify low-energy records in training data
low_train_mask = y_train < low_load_limit

# Extract corresponding input features
X_train_low = X_train.loc[low_train_mask]

# Extract corresponding target values
y_train_low = y_train.loc[low_train_mask]

# Check how many records we selected
print("Low-load training records:", len(y_train_low))


# ==========================================
# EXPERIMENT: LOW-LOAD NEURAL NETWORK
# ==========================================

# Step 1: Define low-energy threshold
low_load_limit = 75

# Step 2: Select low-load training records
low_train_mask = y_train < low_load_limit

X_train_low = X_train.loc[low_train_mask]
y_train_low = y_train.loc[low_train_mask]

print("\n=== LOW-LOAD TRAINING DATA ===")
print("Low-load training records:", len(y_train_low))


# ==========================================
# STEP 3: SCALE LOW-LOAD TRAINING DATA
# ==========================================

low_scaler = StandardScaler()

X_train_low_scaled = low_scaler.fit_transform(
    X_train_low
)


# ==========================================
# STEP 4: CREATE AND TRAIN NEURAL NETWORK
# ==========================================

low_nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)

low_nn_model.fit(
    X_train_low_scaled,
    y_train_low
)


# ==========================================
# STEP 5: SELECT LOW-LOAD TESTING RECORDS
# ==========================================

low_test_mask = y_test < low_load_limit

X_test_low = X_test.loc[low_test_mask]
y_test_low = y_test.loc[low_test_mask]

print("Low-load testing records:", len(y_test_low))


# ==========================================
# STEP 6: SCALE LOW-LOAD TESTING DATA
# ==========================================

X_test_low_scaled = low_scaler.transform(
    X_test_low
)


# ==========================================
# STEP 7: MAKE PREDICTIONS WITH THREE MODELS
# ==========================================

# Original linear regression
lr_low_pred = model.predict(X_test_low)

# Original neural network
original_nn_low_pred = nn_model.predict(
    scaler.transform(X_test_low)
)

# New low-load neural network
low_nn_pred = low_nn_model.predict(
    X_test_low_scaled
)


# ==========================================
# STEP 8: EVALUATE ALL THREE MODELS
# ==========================================

print("\n=== LOW-LOAD MODEL COMPARISON ===")

models = {
    "Linear Regression": lr_low_pred,
    "Original Neural Network": original_nn_low_pred,
    "Low-Load Neural Network": low_nn_pred
}

for name, predictions in models.items():

    mae = mean_absolute_error(
        y_test_low,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_low, predictions)
    )

    r2 = r2_score(
        y_test_low,
        predictions
    )

    print("\n", name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 4))


# ==========================================
# STEP 9: COMPARE PREDICTION ERRORS
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(11, 6))

plt.scatter(
    y_test_low,
    lr_low_pred - y_test_low,
    alpha=0.3,
    label="Linear Regression"
)

plt.scatter(
    y_test_low,
    original_nn_low_pred - y_test_low,
    alpha=0.3,
    label="Original Neural Network"
)

plt.scatter(
    y_test_low,
    low_nn_pred - y_test_low,
    alpha=0.3,
    label="Low-Load Neural Network"
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--"

)

plt.xlabel("Actual Chiller Energy (kWh)")
plt.ylabel("Prediction Error (kWh)")
plt.title("Miami Hotel — Low-Load Model Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("chiller_low_load_comparison.png", dpi=300)
plt.show()


# ==========================================
# LOW-LOAD NEURAL NETWORK
# ACTUAL VS PREDICTED ENERGY
# ==========================================

plt.figure(figsize=(10, 7))

# Plot actual vs predicted values
plt.scatter(
    y_test_low,
    low_nn_pred,
    alpha=0.4,
    color="green",
    label="Low-Load Neural Network"
)

# Perfect prediction reference line
plt.plot(
    [0, 75],
    [0, 75],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Chiller Energy (kWh)")
plt.ylabel("Predicted Chiller Energy (kWh)")

plt.title(
    "Miami Hotel - Low-Load Neural Network\n"
    "Actual vs Predicted Energy"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "low_load_actual_vs_predicted.png",
    dpi=300
)

plt.show()



import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# LOW-LOAD LINEAR REGRESSION
# ACTUAL VS PREDICTED ENERGY
# ==========================================



actual = np.asarray(y_test).ravel()
predicted = np.asarray(y_pred).ravel()

# Select the SAME low-load test records
low_mask = actual < 75

actual_low = actual[low_mask]
predicted_low = predicted[low_mask]

# Create graph
plt.figure(figsize=(10, 7))

plt.scatter(
    actual_low,
    predicted_low,
    alpha=0.4,
    color="blue",
    label="Linear Regression"
)

# Perfect prediction reference line
plt.plot(
    [0, 75],
    [0, 75],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Chiller Energy (kWh)")
plt.ylabel("Predicted Chiller Energy (kWh)")

plt.title(
    "Miami Hotel - Low-Load Linear Regression\n"
    "Actual vs Predicted Energy"
)

# Same scale for direct comparison
plt.xlim(0, 75)
plt.ylim(-25, 125)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "low_load_linear_regression_actual_vs_predicted.png",
    dpi=300
)

plt.show()