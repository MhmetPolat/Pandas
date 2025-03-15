import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Netflix veri setini yükleme
df = pd.read_csv("datasets/netflix_titles.csv")

# İlk 5 satırı görüntüleme
print(df.head())

# Eksik değerleri kontrol etme
missing_values = df.isnull().sum()
print("Eksik değerler:")
print(missing_values[missing_values > 0])

# Eksik değerleri doldurma
df.fillna({"director": "Unknown", "cast": "Unknown", "country": "Unknown", "rating": "Not Rated"}, inplace=True)

# Tarih sütunlarını dönüştürme
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')

# Veri standardizasyonu - Tüm harfleri küçük yapma
df['title'] = df['title'].str.lower()

# Normalizasyon - Yayın sürelerini normalize etme
df['duration'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['duration'] = (df['duration'] - df['duration'].min()) / (df['duration'].max() - df['duration'].min())

# GroupBy kullanımı - İçerik türüne göre sayım
type_counts = df.groupby("type")["show_id"].count()
print(type_counts)

# Filtreleme - Belirli bir ülkeye ait içerikleri seçme
us_movies = df.query("country == 'United States' & type == 'Movie'")

# Veri görselleştirme - İçerik türlerinin dağılımı
sns.countplot(data=df, x="type")
plt.title("Netflix İçerik Türleri Dağılımı")
plt.show()

# Merge ve Join işlemleri için sahte bir ek veri seti oluşturma
dummy_data = pd.DataFrame({
    "show_id": df["show_id"].sample(100),
    "rating_score": np.random.uniform(1, 10, 100)
})

# Veri birleştirme (merge)
df_merged = df.merge(dummy_data, on="show_id", how="left")

# Düzenlenmiş veri setini kaydetme
df.to_csv("cleaned_netflix_data.csv", index=False)
print("Temizlenmiş veri kaydedildi.")