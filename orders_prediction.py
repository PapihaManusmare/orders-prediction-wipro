import pandas as pd
import numpy as np


np.random.seed(42)


dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

df = pd.DataFrame({
    'date': dates,
    'num_orders': np.random.randint(50, 300, size=len(dates)),
    'price': np.random.uniform(10, 500, size=len(dates)),
    'discount': np.random.choice([0, 5, 10, 20], size=len(dates)),
    'product_category': np.random.choice(
                        ['Electronics', 'Clothing', 'Food'],
                        size=len(dates)),
    'campaign_status': np.random.choice(['Yes', 'No'], size=len(dates))
})


df.to_csv('orders.csv', index=False)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())





print("\n--- Basic Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Unique Categories ---")
print("Product Categories:", df['product_category'].unique())
print("Campaign Status:", df['campaign_status'].unique())


print("\n--- Order Stats ---")
print("Min Orders:", df['num_orders'].min())
print("Max Orders:", df['num_orders'].max())
print("Average Orders:", round(df['num_orders'].mean(), 2))


from sklearn.preprocessing import StandardScaler

df['campaign_status'] = df['campaign_status'].map({'Yes': 1, 'No': 0})
print("\n--- Campaign Status Encoded ---")
print(df['campaign_status'].value_counts())

df = pd.get_dummies(df, columns=['product_category'], dtype=int)
print("\n--- After Encoding Categories ---")
print(df.columns.tolist())

scaler = StandardScaler()
df[['price', 'discount']] = scaler.fit_transform(df[['price', 'discount']])
print("\n--- After Normalizing Price & Discount ---")
print(df[['price', 'discount']].head())

print("\n--- Missing Values After Preprocessing ---")
print(df.isnull().sum())


print("\n--- Final Shape ---")
print(df.shape)

print("\n--- Final Columns ---")
print(df.columns.tolist())




import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))
plt.plot(df['date'], df['num_orders'], color='blue', linewidth=0.8)
plt.title('Number of Orders Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('orders_over_time.png')
plt.show()
print("Graph 1 saved ")

df['month'] = df['date'].dt.month
monthly = df.groupby('month')['num_orders'].mean()

plt.figure(figsize=(10, 4))
monthly.plot(kind='bar', color='orange')
plt.title('Average Orders by Month')
plt.xlabel('Month')
plt.ylabel('Average Orders')
plt.tight_layout()
plt.savefig('orders_by_month.png')
plt.show()
print("Graph 2 saved ")

df['day_of_week'] = df['date'].dt.dayofweek
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
weekly = df.groupby('day_of_week')['num_orders'].mean()

plt.figure(figsize=(10, 4))
weekly.plot(kind='bar', color='green')
plt.title('Average Orders by Day of Week')
plt.xticks(ticks=range(7), labels=days, rotation=0)
plt.xlabel('Day')
plt.ylabel('Average Orders')
plt.tight_layout()
plt.savefig('orders_by_day.png')
plt.show()
print("Graph 3 saved ")

plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Graph 4 saved ")



df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print("\n--- Time Features Added ---")
print(df[['date', 'day_of_week', 'month', 
          'day_of_month', 'quarter', 'is_weekend']].head())

df['lag_1'] = df['num_orders'].shift(1)   # yesterday
df['lag_7'] = df['num_orders'].shift(7)   # last week
df['lag_30'] = df['num_orders'].shift(30) # last month

print("\n--- Lag Features Added ---")
print(df[['date', 'num_orders', 'lag_1', 
          'lag_7', 'lag_30']].head(10))

df['rolling_mean_7'] = df['num_orders'].rolling(7).mean()
df['rolling_mean_30'] = df['num_orders'].rolling(30).mean()

print("\n--- Rolling Average Features Added ---")
print(df[['date', 'num_orders', 
          'rolling_mean_7', 'rolling_mean_30']].head(35))

df.dropna(inplace=True)
print("\n--- Shape After Dropping NaN rows ---")
print(df.shape)

print("\n--- All Final Columns ---")
print(df.columns.tolist())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X = df.drop(['num_orders', 'date'], axis=1)
y = df['num_orders']

print("\n--- Features Shape ---")
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("\n--- Train/Test Split ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


print("\n--- Training Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression Trained ")


print("\n--- Training Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Trained ")


print("\n--- Training XGBoost ---")
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
print("XGBoost Trained ")

print("\n--- All Models Trained Successfully! ---")



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("=" * 45)

for name, preds in [('Linear Regression', lr_preds),
                    ('Random Forest',     rf_preds),
                    ('XGBoost',           xgb_preds)]:

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.4f}")

print("\n" + "=" * 45)

results = {
    'Linear Regression': r2_score(y_test, lr_preds),
    'Random Forest':     r2_score(y_test, rf_preds),
    'XGBoost':           r2_score(y_test, xgb_preds)
}

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model}")
print(f" Best R² Score: {results[best_model]:.4f}")


plt.figure(figsize=(14, 5))
plt.plot(y_test.values, label='Actual Orders', 
         color='blue', linewidth=1)
plt.plot(lr_preds,  label='Linear Regression', 
         color='red', linestyle='--', linewidth=1)
plt.plot(rf_preds,  label='Random Forest', 
         color='green', linestyle='--', linewidth=1)
plt.plot(xgb_preds, label='XGBoost', 
         color='orange', linestyle='--', linewidth=1)
plt.title('Actual vs Predicted Orders - All Models')
plt.xlabel('Test Data Points')
plt.ylabel('Number of Orders')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_all.png')
plt.show()
print("Graph 5 saved ")


plt.figure(figsize=(14, 5))
plt.plot(y_test.values, label='Actual Orders', 
         color='blue', linewidth=1.5)
plt.plot(lr_preds, label='Predicted Orders (Linear Regression)', 
         color='red', linestyle='--', linewidth=1.5)
plt.title('Actual vs Predicted Orders - Best Model')
plt.xlabel('Test Data Points')
plt.ylabel('Number of Orders')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_best.png')
plt.show()
print("Graph 6 saved ")

plt.figure(figsize=(10, 6))
feat_importance = pd.Series(
    rf.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

feat_importance.plot(kind='bar', color='purple')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Graph 7 saved ")

print("\n--- All Visualizations Done! ---")



import joblib

joblib.dump(lr, 'best_model.pkl')
print("\n--- Model Saved ---")
print("best_model.pkl saved ✅")

joblib.dump(scaler, 'scaler.pkl')
print("scaler.pkl saved ✅")


loaded_model = joblib.load('best_model.pkl')
test_preds = loaded_model.predict(X_test)
print("\n--- Model Loaded & Tested Successfully ---")
print("Sample Predictions:", test_preds[:5].round(2))
print("Actual Values:     ", y_test.values[:5])

print("\n" + "=" * 45)
print("PROJECT COMPLETE!")
print("=" * 45)
print("Files saved in your project folder:")
print("  orders.csv                  - Dataset")
print("  best_model.pkl              - Saved Model")
print("  scaler.pkl                  - Saved Scaler")
print("  orders_over_time.png        - Graph 1")
print("  orders_by_month.png         - Graph 2")
print("  orders_by_day.png           - Graph 3")
print("  correlation_heatmap.png     - Graph 4")
print("  actual_vs_predicted_all.png - Graph 5")
print("  actual_vs_predicted_best.png- Graph 6")
print("  feature_importance.png      - Graph 7")
print("=" * 45)