import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("AB_NYC_2019.csv")

df = df[['price', 'minimum_nights', 'availability_365', 'number_of_reviews', 'room_type', 'neighbourhood_group']]
df = df[df['price'] <= 5000]

df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood_group'], drop_first=True)

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

model = LinearRegression()
model.fit(X, y)

with open('airbnb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model başarıyla eğitildi.")
print("📌 Kullanılan özellikler:")
print(X.columns.tolist())
