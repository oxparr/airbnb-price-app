# 1. VERİYİ YÜKLE VE İLK ANALİZLERİ YAP
# Gerekli kütüphaneleri yükle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oku
file_path = 'AB_NYC_2019.csv'
df = pd.read_csv(file_path)

# Veri seti hakkında genel bilgi
print('--- Veri Seti Bilgisi ---')
df.info()

print('\n--- İstatistiksel Özet ---')
print(df.describe())

# Eksik değer analizi
print('\n--- Eksik Değerler ---')
print(df.isnull().sum())

# Eksik değer içeren sütunları gerekirse sil
# (last_review, reviews_per_month, host_name, name gibi)
df = df.drop(['last_review', 'reviews_per_month', 'host_name', 'name'], axis=1)
print('\nGüncel sütunlar:', df.columns.tolist())

# 2. EDA (EXPLORATORY DATA ANALYSIS)

# Ortalama fiyatı neighbourhood_group ve room_type'a göre hesapla
plt.figure(figsize=(10,5))
sns.barplot(data=df, x='neighbourhood_group', y='price', ci=None)
plt.title('Neighbourhood Group Bazında Ortalama Fiyat')
plt.ylabel('Ortalama Fiyat')
plt.xlabel('Neighbourhood Group')
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(data=df, x='room_type', y='price', ci=None)
plt.title('Oda Tipine Göre Ortalama Fiyat')
plt.ylabel('Ortalama Fiyat')
plt.xlabel('Oda Tipi')
plt.show()

# Price dağılımı
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Fiyat Dağılımı')
plt.xlabel('Fiyat')
plt.ylabel('Frekans')
plt.show()

# Sayısal değişkenlerle price arasındaki ilişki
num_cols = ['number_of_reviews', 'availability_365']
if 'reviews_per_month' in df.columns:
    num_cols.append('reviews_per_month')

for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.scatterplot(data=df, x=col, y='price', alpha=0.3)
    plt.title(f'{col} vs Price')
    plt.show()

# Korelasyon matrisi ve heatmap
df_corr = df[['price'] + num_cols].corr()
plt.figure(figsize=(6,4))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

# Outlier (uç değer) analizi
plt.figure(figsize=(8,5))
sns.histplot(df[df['price'] > 5000]['price'], bins=30, color='red')
plt.title('5000 Üzerindeki Fiyatlar (Outlier Analizi)')
plt.xlabel('Fiyat')
plt.ylabel('Frekans')
plt.show()

# Uç değerleri filtrele (price > 5000 olanları çıkar)
df = df[df['price'] <= 5000]
print(f"\nUç değerler çıkarıldıktan sonra kalan veri seti boyutu: {df.shape}")

# 3. VERİ HAZIRLAMA
# Kullanılmayacak sütunları sil: id, name, host_name, last_review (zaten silindi)
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Kategorik değişkenleri OneHotEncoding ile sayısallaştır
cat_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Hedef değişken ve özellikler
X = df.drop('price', axis=1)
y = df['price']

# Veriyi eğitim ve test olarak ayır
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim veri seti boyutu: {X_train.shape}")
print(f"Test veri seti boyutu: {X_test.shape}")

# 4. MODELLEME
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Sonuçları saklamak için bir tablo
results = []

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"\n{name} Sonuçları:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.3f}")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate_model('Linear Regression', y_test, y_pred_lr)

# Decision Tree Regressor (max_depth=5, 10, None)
for depth in [5, 10, None]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    name = f"Decision Tree (max_depth={depth})"
    evaluate_model(name, y_test, y_pred_dt)

# Lasso Regression (alpha=0.1, 1, 10)
for alpha in [0.1, 1, 10]:
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    name = f"Lasso Regression (alpha={alpha})"
    evaluate_model(name, y_test, y_pred_lasso)

# 5. MODEL KARŞILAŞTIRMA
print('\n--- Model Karşılaştırma Tablosu ---')
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)

# En iyi modeli bul (en yüksek R2)
best_model = results_df.loc[results_df['R2'].idxmax()]
print(f"\nEn iyi model: {best_model['Model']} (R2: {best_model['R2']:.3f})")

# Lasso Regression ile hangi değişkenlerin etkisi sıfırlandı?
# (alpha=1 ile eğitilen son Lasso modelini kullanıyoruz)
last_lasso = Lasso(alpha=1, max_iter=10000, random_state=42)
last_lasso.fit(X_train, y_train)
zero_coef_features = X_train.columns[last_lasso.coef_ == 0].tolist()
print(f"\nLasso Regression (alpha=1) ile etkisi sıfırlanan değişkenler:")
print(zero_coef_features)

# Açıklama:
print("\nYorum: Lasso Regression, bazı değişkenlerin katsayısını sıfırlayarak modelde önemli olmayanları otomatik olarak elemiş olur. Sıfırlanan değişkenler yukarıda listelenmiştir.")

# 6. SONUÇLAR VE YORUMLAMA
print('\n--- SONUÇLAR VE YORUMLAMA ---')

# En önemli değişkenleri bulmak için Linear Regression katsayılarını kullanabiliriz
def print_top_features(model, X, n=10):
    coefs = pd.Series(model.coef_, index=X.columns)
    top = coefs.abs().sort_values(ascending=False).head(n)
    print(f"\nFiyatı en çok etkileyen {n} değişken:")
    print(top)

print_top_features(lr, X_train)

print("\nModelin genel başarısı: ")
print(f"Linear Regression R2: {results_df.loc[results_df['Model']=='Linear Regression', 'R2'].values[0]:.3f}")
print(f"Decision Tree (max_depth=10) R2: {results_df.loc[results_df['Model']=='Decision Tree (max_depth=10)', 'R2'].values[0]:.3f}")
print(f"Lasso Regression (alpha=1) R2: {results_df.loc[results_df['Model']=='Lasso Regression (alpha=1)', 'R2'].values[0]:.3f}")

print("\nYorum:")
print("- Fiyatı en çok etkileyen değişkenler oda tipi, konum (özellikle Manhattan ve Brooklyn), minimum_nights gibi değişkenlerdir.")
print("- Modelin başarısı, veri setindeki yüksek varyans ve outlier'lar nedeniyle sınırlı olabilir. Özellikle fiyat dağılımı sağa çarpık olduğu için RMSE yüksek çıkabilir.")
print("- Manhattan gibi bölgelerde fiyatlar, talebin yüksekliği, merkezi konum, turistik cazibe ve lüks konaklama seçenekleri nedeniyle yüksektir.")

# (OPSİYONEL BONUS)
print('\n--- BONUS: Feature Importance, GridSearchCV ve Model Kaydetme ---')

# Feature Importance - Decision Tree
best_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
best_dt.fit(X_train, y_train)
importances = pd.Series(best_dt.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10,6))
importances.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title('Decision Tree Feature Importances (Top 15)')
plt.ylabel('Önem Skoru')
plt.show()

# Feature Importance - Lasso
lasso = Lasso(alpha=1, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)
lasso_importances = pd.Series(np.abs(lasso.coef_), index=X_train.columns)
plt.figure(figsize=(10,6))
lasso_importances.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title('Lasso Feature Importances (Top 15)')
plt.ylabel('Katsayının Mutlak Değeri')
plt.show()

# GridSearchCV ile Decision Tree için en iyi parametreleri bul
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [3, 5, 7, 10, 15, None], 'min_samples_split': [2, 5, 10]}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs.fit(X_train, y_train)
print(f"\nGridSearchCV ile en iyi Decision Tree parametreleri: {gs.best_params_}")
print(f"En iyi skor (neg MSE): {gs.best_score_}")

# Modeli pickle ile kaydet
import pickle
with open('airbnb_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("\nLinear Regression modeli 'airbnb_model.pkl' olarak kaydedildi.")
