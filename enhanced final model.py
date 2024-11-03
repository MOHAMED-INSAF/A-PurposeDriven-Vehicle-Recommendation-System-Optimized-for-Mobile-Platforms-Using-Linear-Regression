import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import messagebox


dataset_path = 'vehicles_dataset.csv'
vehicles_df = pd.read_csv(dataset_path)

# Encode non-numeric data for the model
vehicles_df['fuel'] = LabelEncoder().fit_transform(vehicles_df['fuel'])
vehicles_df['body'] = LabelEncoder().fit_transform(vehicles_df['body'])
vehicles_df['Purpose'] = vehicles_df['Purpose'].map({'Budget': 0, 'Luxury': 1})

# Define a function to categorize vehicles by their purpose
def categorize_purpose(row):
    # Urban: Budget cars with better mileage
    if row['Purpose'] == 0 and row['mileage'] >= 25:
        return 'Urban'
    # Touring: Balanced for long drives with decent mileage
    elif row['Purpose'] == 0 and row['mileage'] < 25:
        return 'Touring'
    # Racing: High performance, more cylinders, often in the luxury range
    elif row['Purpose'] == 1 and row['cylinders'] >= 6:
        return 'Racing'
    # Default to Touring if no specific match
    else:
        return 'Touring'

# Apply the categorization function to the dataset
vehicles_df['Detailed_Purpose'] = vehicles_df.apply(categorize_purpose, axis=1)
# Map purpose categories to numerical values for modeling
vehicles_df['Detailed_Purpose'] = vehicles_df['Detailed_Purpose'].map({'Urban': 0, 'Touring': 1, 'Racing': 2})

# Set up features and target for the model
feature_columns = ['price', 'mileage', 'fuel', 'body', 'Purpose', 'Detailed_Purpose', 'cylinders']
X = vehicles_df[feature_columns]
y = vehicles_df['mileage']  # Mileage is used as a proxy for the suitability score

# Fill missing values and standardize the data
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
y = y.fillna(y.mean())

# Standardize features to bring them to a similar scale
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

# Train the Linear Regression Model with cross-validation for better robustness
model = LinearRegression()
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
# Using negative MSE because cross_val_score returns negative for minimization problems
cross_val_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {[-score for score in cross_val_scores]}")
print(f"Average Cross-Validation MSE: {(-cross_val_scores.mean())}")

# Train the model on the entire dataset for the GUI application
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"Model MSE: {mse}, MAE: {mae}")

# Define how to calculate suitability based on user's intended use
def calculate_purpose_score(row, user_purpose):
    # Higher weights for mileage and price for Urban, considering affordability and efficiency
    if user_purpose == 'Urban':
        return (0.6 * row['mileage']) + (0.4 / row['price'])
    # Balance mileage and fuel efficiency for Touring vehicles
    elif user_purpose == 'Touring':
        return (0.5 * row['mileage']) + (0.3 / row['price']) + (0.2 * row['fuel'])
    # Performance focus for Racing: prioritize cylinders and body type
    elif user_purpose == 'Racing':
        return (0.4 * row['cylinders']) + (0.4 * row['body']) + (0.2 * row['price'])
    return 0  # Default fallback, though this shouldn't be reached

# Generate recommendations based on user input
def recommend_vehicles(user_budget, user_mileage_preference, purpose, top_n=5):
    purpose_map = {'Urban': 0, 'Touring': 1, 'Racing': 2}
    purpose_code = purpose_map.get(purpose, 1)  # Default to Touring if purpose is unclear

    # Filter vehicles based on user preferences
    filtered_df = vehicles_df[
        (vehicles_df['price'] <= user_budget) & 
        (vehicles_df['mileage'] >= user_mileage_preference) & 
        (vehicles_df['Detailed_Purpose'] == purpose_code)
    ]

    # Score the vehicles based on their suitability for the selected purpose
    filtered_df['suitability_score'] = filtered_df.apply(lambda row: calculate_purpose_score(row, purpose), axis=1)
    # Sort by suitability and return the top matches
    recommended_vehicles = filtered_df.sort_values(by='suitability_score', ascending=False).head(top_n)
    return recommended_vehicles[['name', 'price', 'mileage', 'fuel', 'body', 'Purpose', 'Detailed_Purpose', 'suitability_score']]

# Build the user interface using Tkinter
def show_recommendations():
    try:
        user_budget = float(budget_entry.get())
        user_mileage_preference = float(mileage_entry.get())
        user_purpose = purpose_var.get()

        # Get recommendations and display them
        recommendations = recommend_vehicles(user_budget, user_mileage_preference, user_purpose)
        
        if not recommendations.empty:
            result = "Top Recommendations Based on Your Input:\n\n"
            for idx, row in recommendations.iterrows():
                purpose_text = {0: "Urban", 1: "Touring", 2: "Racing"}[row['Detailed_Purpose']]
                result += f"{row['name']} - Price: ${row['price']}, Mileage: {row['mileage']} MPG, Purpose: {purpose_text} ({'Budget' if row['Purpose'] == 0 else 'Luxury'})\n"
            messagebox.showinfo("Vehicle Recommendations", result)
        else:
            messagebox.showinfo("No Results", "No vehicles match your criteria.")
    
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for budget and mileage.")

# Initialize the interface window
root = tk.Tk()
root.title("Vehicle Recommendation System")

# Input fields for user preferences
tk.Label(root, text="Enter Your Budget ($):").grid(row=0, column=0, padx=10, pady=5)
budget_entry = tk.Entry(root)
budget_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Preferred Mileage (MPG):").grid(row=1, column=0, padx=10, pady=5)
mileage_entry = tk.Entry(root)
mileage_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Purpose:").grid(row=2, column=0, padx=10, pady=5)
purpose_var = tk.StringVar(value="Urban")
purpose_menu = tk.OptionMenu(root, purpose_var, "Urban", "Touring", "Racing")
purpose_menu.grid(row=2, column=1, padx=10, pady=5)

# Button to get recommendations
submit_button = tk.Button(root, text="Get Recommendations", command=show_recommendations)
submit_button.grid(row=3, column=0, columnspan=2, pady=10)

# Start the Tkinter main loop
root.mainloop()
