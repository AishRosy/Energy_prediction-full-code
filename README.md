# Energy_prediction-full-code
Load dataset Preprocessing Add feature Drop unused column Splitting dataset  Preprocessing pipeline Define models with hyperparameter tuning Save prediction Calculate evaluation matrix
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations and array manipulations
import matplotlib.pyplot as plt  # Data visualization and plotting
import seaborn as sns  # Statistical data visualization
from sklearn.model_selection import train_test_split, GridSearchCV  # Splitting data and hyperparameter tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Model evaluation metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # Regression models
from sklearn.svm import SVR  # Support Vector Regression model
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Preprocessing for numerical and categorical data
from sklearn.compose import ColumnTransformer  # Handling transformations for mixed data types
from sklearn.pipeline import Pipeline  # Creating pipelines for preprocessing and modeling
from datetime import datetime  # Working with date and time
from sklearn.feature_selection import SelectKBest, f_regression  # Feature selection methods
import scipy.stats as stats  # Statistical functions

# Load dataset
energy_data = pd.read_csv("energydata_complete.csv")  # Loading the dataset to analyze energy consumption patterns

# Convert date column to datetime
energy_data['date'] = pd.to_datetime(energy_data['date'], format='%Y-%m-%d %H:%M:%S')  # Converting date column to datetime format

# Add features
energy_data['NSM'] = energy_data['date'].dt.hour * 3600 + energy_data['date'].dt.minute * 60 + energy_data['date'].dt.second  # Calculating seconds since midnight to capture time-based trends
energy_data['WeekStatus'] = energy_data['date'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')  # Classifying days as weekdays or weekends
energy_data['Day_of_week'] = energy_data['date'].dt.day_name()  # Extracting the name of the day to analyze patterns by day

# Drop unused columns
energy_data = energy_data.drop(['date'], axis=1)  # Dropping the original date column as it's no longer needed

# Splitting the dataset
X = energy_data.drop('Appliances', axis=1)  # Defining features
y = energy_data['Appliances']  # Defining target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # Splitting data into training and testing sets

# Preprocessing pipeline
categorical_features = ['WeekStatus', 'Day_of_week']  # List of categorical features
numerical_features = [col for col in X.columns if col not in categorical_features]  # Identifying numerical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Standardizing numerical features
        ('cat', OneHotEncoder(), categorical_features)  # Encoding categorical features using one-hot encoding
    ]
)

# Define models with hyperparameter tuning
param_grid = {
    'GradientBoosting': {
        'model__n_estimators': [100, 200, 300],  # Number of boosting stages
        'model__max_depth': [3, 5, 7],  # Depth of trees
        'model__learning_rate': [0.05, 0.1, 0.2]  # Learning rate for boosting
    },
    'RandomForest': {
        'model__n_estimators': [100, 200, 300],  # Number of trees in the forest
        'model__max_depth': [5, 10, 15]  # Maximum depth of trees
    },
    'SVR': {
        'model__C': [0.1, 1, 10],  # Regularization parameter
        'model__epsilon': [0.01, 0.1, 0.5]  # Epsilon-tube within which predictions are considered correct
    }
}

results = {}  # Dictionary to store results for each model
predictions = pd.DataFrame()  # DataFrame to store predictions
for name, param in param_grid.items():
    # Initialize the model based on its name
    if name == 'GradientBoosting':
        model = GradientBoostingRegressor()
    elif name == 'RandomForest':
        model = RandomForestRegressor()
    else:
        model = SVR(kernel='rbf')

    # Create a pipeline for preprocessing, feature selection, and modeling
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocess data (scaling and encoding)
        ('selector', SelectKBest(score_func=f_regression, k=10)),  # Feature selection after preprocessing
        ('model', model)  # Apply the regression model
    ])
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid=param, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)  # Fit the model using grid search
    y_pred = grid_search.best_estimator_.predict(X_test)  # Make predictions on test data

    # Save predictions
    predictions[name] = y_pred

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R-squared score

predictions.to_csv('predictions.csv', index=False)  # Save predictions to CSV

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(18, 5))  # Create subplots for metrics visualization
for i, (name, metrics) in enumerate(results.items()):
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax[i])  # Plot metrics for each model
    ax[i].set_title(name)  # Add title for each plot
plt.tight_layout()
plt.show()

# Correlation matrix and heatmap
X_encoded = pd.get_dummies(X_train, columns=['WeekStatus', 'Day_of_week'])  # Encode categorical variables
corr_matrix = X_train.corr()  # Calculate correlations between features
plt.figure(figsize=(12, 8))  # Set plot size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  # Plot correlation matrix as a heatmap
plt.title('Correlation Matrix')  # Add title
plt.show()

# Save training and testing sets
X_train.to_csv('training.csv', index=False)  # Save training set to CSV file
X_test.to_csv('testing.csv', index=False)  # Save testing set to CSV file

print("Training, testing, and prediction files saved successfully.")    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}  # Store results
    print(f"{name} - Best Params: {grid_search.best_params_} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")  # Print results

# Save actual vs predicted results
predictions['Actual'] = y_test.values

