import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import numpy as np

# 1. Veri Yükle
df = pd.read_csv("AB_NYC_2019.csv")
df = df[['price', 'minimum_nights', 'availability_365', 'number_of_reviews', 'room_type', 'neighbourhood_group']]
df = df[df['price'] <= 5000]
df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood_group'], drop_first=True)

# 2. Veriyi ayır
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Parametre aralığını belirle
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# 4. GridSearchCV ile en iyi modeli bul
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

# 5. En iyi modeli al
best_model = grid.best_estimator_

# 6. Performans değerlendirmesi
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ En iyi model parametreleri:", grid.best_params_)
print(f"📊 MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

# 7. Kaydet
with open("airbnb_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
