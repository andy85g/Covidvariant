Epidemic Duration Prediction Using Flask and Machine Learning

Overview

This project is a Flask-based web application that analyzes COVID-19 data and builds a machine-learning model to predict the duration of an epidemic. The project involves data collection, cleaning, visualization, feature engineering, model training, and evaluation.

Features

Data collection and preprocessing

Exploratory data analysis and visualization

Machine learning model to predict epidemic duration

Feature importance analysis

Flask web application for data interaction and prediction

Technologies Used
1.Flask
2. Pandas
3. Matplotlib
4. Seaborn
5. SHAP
6. Joblib
7.GeoPandas
8.NumPy
9.Scipy
10.Scikit-learn (RandomForestRegressor, GradientBoostingRegressor, KMeans, etc.)

Dataset

The dataset consists of COVID-19 statistics from different countries, including:

Country – Country name
first_seq – First sequence date
num_seqs – Number of sequences
last_seq – Last sequence date
variant – Variant type
censure_date – Censure date
duration – Duration of epidemic
censored – Whether the data is censored
mortality_rate – Mortality rate
total_cases – Total cases
total_deaths – Total deaths
growth_rate – Growth rate
Installation

Clone the repository:

git clone https://github.com/your-repo.git
cd your-repo
Install dependencies:
pip install -r requirements.txt

Run the Flask application:
python app.py

Data Processing
Data Cleaning
Handling missing values using forward fill (fillna(method='ffill'))

Removing duplicate records

Saving the cleaned dataset as cleaned_covid_data.csv

Feature Engineering

Calculating mortality_case_ratio as total_deaths / (total_cases + 1)

Applying log transformation to total_cases and total_deaths

Selecting relevant features: num_seqs, mortality_case_ratio, total_cases, total_deaths, growth_rate, censored

Model Training

Splitting Data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Scaling Data:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Training RandomForest Model:

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

Evaluating Model:

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")

Prediction Example

Predicting epidemic duration for new data:

new_data = pd.DataFrame({
    'num_seqs': [5000],
    'mortality_case_ratio': [0.02],  
    'total_cases': [10000],  
    'total_deaths': [200],  
    'growth_rate': [0.5],  
    'censored': [1]
})
new_data_scaled = scaler.transform(new_data)
predicted_duration = model.predict(new_data_scaled)
print(f"Predicted Epidemic Duration: {predicted_duration[0]} days")

Feature Importance

Analyzing which features contribute the most to the prediction:

feature_importances = pd.Series(model.feature_importances_, index=features)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel("Feature Importance Score")
plt.title("Important Features for Epidemic Duration Prediction")
plt.show()








  


