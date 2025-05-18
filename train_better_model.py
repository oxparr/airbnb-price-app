import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Veriyi oku
df = pd.read_csv("AB_NYC_2019.csv")

# 2. Yalnızca gerekli sütunlar
df = df[['price', 'minimum_nights', 'availability_365', 'number_of_reviews', 'room_type', 'neighbourhood_group']]
df = df[df['price'] <= 5000]

# 3. One-hot encoding
df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood_group'], drop_first=True)

# 4. Özellik ve hedef değişken
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 5. Modeli eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Kaydet
with open('airbnb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Random Forest modeli başarıyla eğitildi ve kaydedildi.")
print("📌 Kullanılan özellikler:", X.columns.tolist())
