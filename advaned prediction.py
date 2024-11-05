import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import messagebox, Toplevel, Text, Scrollbar, RIGHT, Y

# Load the dataset
dataset_path = 'vehicles_dataset.csv'
vehicles_df = pd.read_csv(dataset_path)

# Round price values to the nearest integer for easier interpretation
vehicles_df['price'] = vehicles_df['price'].round(0).astype(int)

# Encode non-numeric columns to prepare for machine learning, 
# storing mappings for future display in the GUI.
label_encoders = {}
for col in ['fuel', 'body', 'Purpose', 'Detailed_Purpose', 'drivetrain']:
    le = LabelEncoder()
    vehicles_df[col] = le.fit_transform(vehicles_df[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Save the original prices for display (we'll use scaled prices only for calculations)
vehicles_df['original_price'] = vehicles_df['price']

# Define feature columns for model training and the target variable
feature_columns = ['price', 'mileage', 'fuel', 'body', 'Purpose', 
                   'Detailed_Purpose', 'cylinders', 'Displacement (cc)', 'drivetrain']
X = vehicles_df[feature_columns]
y = vehicles_df['mileage']

# Handle missing values and standardize the data for model input
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
y = y.fillna(y.median())

# Standardize features (but keep the unscaled original prices for display)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[feature_columns] = scaler.fit_transform(X[feature_columns])
vehicles_df['scaled_price'] = X_scaled['price']

# Initialize and train the model, checking performance with cross-validation
model = LinearRegression()
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {[-score for score in cross_val_scores]}")
print(f"Average Cross-Validation MSE: {(-cross_val_scores.mean())}")

# Fit the model and calculate errors for reference
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"Model MSE: {mse}, MAE: {mae}")

# Function to calculate a suitability score for each vehicle based on purpose
def calculate_purpose_score(row, user_purpose):
    if user_purpose == 'Urban':
        return (0.5 * row['mileage']) + (0.3 / row['price']) + (0.2 / row['Displacement (cc)'])
    elif user_purpose == 'Touring':
        return (0.4 * row['mileage']) + (0.3 / row['price']) + (0.2 * row['fuel']) + (0.1 * row['Displacement (cc)'])
    elif user_purpose == 'Racing':
        return (0.4 * row['cylinders']) + (0.3 * row['body']) + (0.2 * row['price']) + (0.1 * row['Displacement (cc)'])
    return 0

# Function to recommend vehicles based on user inputs and suitability score
def recommend_vehicles(user_budget_min, user_budget_max, mileage_min, mileage_max, 
                       user_fuel, user_body, user_drivetrain, purpose, top_n=5):
    purpose_map = {'Urban': 0, 'Touring': 1, 'Racing': 2}
    purpose_code = purpose_map.get(purpose, 1)
    
    # Set up conditions for filtering based on user criteria
    conditions = [
        (vehicles_df['price'] >= user_budget_min),
        (vehicles_df['price'] <= user_budget_max),
        (vehicles_df['mileage'] >= mileage_min),
        (vehicles_df['mileage'] <= mileage_max),
        (vehicles_df['Detailed_Purpose'] == purpose_code)
    ]
    
    # Filter based on optional fields if specified
    if user_fuel != "Any":
        conditions.append(vehicles_df['fuel'] == label_encoders['fuel'][user_fuel])
    if user_body != "Any":
        conditions.append(vehicles_df['body'] == label_encoders['body'][user_body])
    if user_drivetrain != "Any":
        conditions.append(vehicles_df['drivetrain'] == label_encoders['drivetrain'][user_drivetrain])
    
    # Apply conditions to filter the dataset
    filtered_df = vehicles_df[conditions[0]]
    for condition in conditions[1:]:
        filtered_df = filtered_df[condition]

    # Calculate suitability score for each vehicle in the filtered list
    filtered_df['suitability_score'] = filtered_df.apply(lambda row: calculate_purpose_score(row, purpose), axis=1)
    recommended_vehicles = filtered_df.sort_values(by='suitability_score', ascending=False).head(top_n)
    return recommended_vehicles[['name', 'original_price', 'mileage', 'fuel', 'body', 'Purpose', 
                                 'Detailed_Purpose', 'drivetrain', 'suitability_score']]

# Function to display recommendations in a scrollable window
def show_recommendations():
    try:
        # Gather user inputs
        user_budget_min = float(budget_min_entry.get())
        user_budget_max = float(budget_max_entry.get())
        mileage_min = float(mileage_min_entry.get())
        mileage_max = float(mileage_max_entry.get())
        user_fuel = fuel_var.get()
        user_body = body_var.get()
        user_drivetrain = drivetrain_var.get()
        user_purpose = purpose_var.get()
        
        # Determine recommendation count
        top_n = len(vehicles_df) if show_all_var.get() == 1 else int(num_recommendations_entry.get())

        # Convert user selections to numeric values
        if user_fuel != "Any":
            user_fuel = label_encoders['fuel'][user_fuel]
        if user_body != "Any":
            user_body = label_encoders['body'][user_body]
        if user_drivetrain != "Any":
            user_drivetrain = label_encoders['drivetrain'][user_drivetrain]

        # Get recommendations
        recommendations = recommend_vehicles(
            user_budget_min, user_budget_max, mileage_min, mileage_max, 
            user_fuel, user_body, user_drivetrain, user_purpose, top_n
        )
        
        # Set up a window to display recommendations
        recommendation_window = Toplevel(root)
        recommendation_window.title("Vehicle Recommendations")
        
        # Scrollable Text widget for recommendations
        text_widget = Text(recommendation_window, wrap='word', width=80, height=20)
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar = Scrollbar(recommendation_window, command=text_widget.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert recommendations into the text widget
        if not recommendations.empty:
            result = "Top Recommendations:\n\n"
            for idx, row in recommendations.iterrows():
                purpose_text = {0: "Urban", 1: "Touring", 2: "Racing"}[row['Detailed_Purpose']]
                result += f"{row['name']} - Price: ${row['original_price']}, Mileage: {row['mileage']} MPG, Purpose: {purpose_text}\n"
            text_widget.insert("1.0", result)
        else:
            text_widget.insert("1.0", "No vehicles match your criteria.")
    
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")

# Set up GUI
root = tk.Tk()
root.title("Enhanced Vehicle Recommendation System")

# User input fields and labels for budget, mileage, fuel type, etc.
tk.Label(root, text="Budget Range ($):").grid(row=0, column=0, padx=10, pady=5)
budget_min_entry = tk.Entry(root)
budget_min_entry.grid(row=0, column=1, padx=5)
budget_max_entry = tk.Entry(root)
budget_max_entry.grid(row=0, column=2, padx=5)

tk.Label(root, text="Mileage Range (MPG):").grid(row=1, column=0, padx=10, pady=5)
mileage_min_entry = tk.Entry(root)
mileage_min_entry.grid(row=1, column=1, padx=5)
mileage_max_entry = tk.Entry(root)
mileage_max_entry.grid(row=1, column=2, padx=5)

# Additional selection fields
tk.Label(root, text="Fuel Type:").grid(row=2, column=0, padx=10, pady=5)
fuel_var = tk.StringVar(value="Any")
fuel_menu = tk.OptionMenu(root, fuel_var, "Any", *label_encoders['fuel'].keys())
fuel_menu.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Body Type:").grid(row=3, column=0, padx=10, pady=5)
body_var = tk.StringVar(value="Any")
body_menu = tk.OptionMenu(root, body_var, "Any", *label_encoders['body'].keys())
body_menu.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Drivetrain:").grid(row=4, column=0, padx=10, pady=5)
drivetrain_var = tk.StringVar(value="Any")
drivetrain_menu = tk.OptionMenu(root, drivetrain_var, "Any", *label_encoders['drivetrain'].keys())
drivetrain_menu.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Purpose:").grid(row=5, column=0, padx=10, pady=5)
purpose_var = tk.StringVar(value="Urban")
purpose_menu = tk.OptionMenu(root, purpose_var, "Urban", "Touring", "Racing")
purpose_menu.grid(row=5, column=1, padx=10, pady=5)

# Input for specifying the number of recommendations or showing all results
tk.Label(root, text="Number of Recommendations:").grid(row=6, column=0, padx=10, pady=5)
num_recommendations_entry = tk.Entry(root)
num_recommendations_entry.insert(0, "5")  # Default to 5 recommendations
num_recommendations_entry.grid(row=6, column=1, padx=5)

# Checkbox for showing all matching recommendations
show_all_var = tk.IntVar()  # Variable to store checkbox state
show_all_checkbox = tk.Checkbutton(root, text="Show All Recommendations", variable=show_all_var)
show_all_checkbox.grid(row=7, column=0, columnspan=2, pady=5)

# Submit button to fetch recommendations based on user input
submit_button = tk.Button(root, text="Get Recommendations", command=show_recommendations)
submit_button.grid(row=8, column=0, columnspan=3, pady=10)

# Start the Tkinter main loop to display the GUI
root.mainloop()

