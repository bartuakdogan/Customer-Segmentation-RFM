import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Görsel ayarları
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. VERİYİ OKUMA VE HAZIRLAMA
print("⏳ Veri okunuyor...")
# Encoding hatası olmaması için 'ISO-8859-1' kullanıyoruz
df = pd.read_csv('Online_Retail.csv', encoding='ISO-8859-1')

print(f"Orijinal Veri Boyutu: {df.shape}")

# Veri Temizliği:
# - CustomerID'si olmayanları at (Kime ait olduğunu bilmediğimiz işlemi analiz edemeyiz)
df = df.dropna(subset=['CustomerID'])

# - İadeleri (Negatif Quantity) ve Hatalı Fiyatları Temizle
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print(f"Temizlik Sonrası Veri Boyutu: {df.shape}")

# Tarih formatını düzeltme
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Toplam Tutar (TotalPrice) Hesabı
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 2. RFM METRİKLERİNİN HESAPLANMASI
# Analiz tarihi olarak veri setindeki son günden 1 gün sonrası alınmıştır.
analysis_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

print("\n📊 RFM Metrikleri Hazırlanıyor...")
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days, # Recency: En son kaç gün önce geldi?
    'InvoiceNo': 'nunique',                                  # Frequency: Kaç farklı alışveriş yaptı?
    'TotalPrice': 'sum'                                      # Monetary: Toplam ne kadar harcama yaptı?
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 3. VERİYİ MAKİNE ÖĞRENMESİNE HAZIRLAMA (SCALING)
# K-Means algoritması mesafe temelli olduğu için sayıların büyüklüğü (Scale) yani 'Skala' önemlidir.
# Parasal değerler (10.000) ile Gün sayıları (10) arasındaki uçurumu kapatmak için Log Transformation yapıyoruz.
rfm_log = np.log1p(rfm)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# 4. K-MEANS CLUSTERING (KÜMELEME) MODELİ
# Müşterileri 3 Ana Gruba ayıralım
print("\n🤖 Yapay Zeka (K-Means) Müşterileri Grupluyor...")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(rfm_scaled)

# Etiketlerin Ana Veriye Eklenmesi
rfm['Cluster'] = kmeans.labels_

# 5. SONUÇLARIN YORUMLANMASI
print("\n✅ SEGMENTASYON SONUÇLARI:")
# Her bir kümenin ortalama değerleri
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'Cluster': 'count' # O grupta kaç kişi var?
}).rename(columns={'Cluster': 'Müşteri Sayısı'})

print(cluster_summary)

# 6. GÖRSELLEŞTİRME
# Monetary vs Recency grafiği
plt.figure(figsize=(10,6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Müşteri Segmentleri: Recency vs Monetary')
plt.xlabel('En Son Alışveriş (Gün Önce)')
plt.ylabel('Toplam Harcama')
plt.yscale('log') # Harcamalar çok değişken olduğu için logaritmik eksen
plt.legend(title='Segment (Cluster)')
plt.show()

print("\n💡 İPUCU: Hangi Cluster (0, 1 veya 2) en değerli? Frequency ve Monetary'si YÜKSEK, Recency'si DÜŞÜK olan grup senin 'Şampiyon' müşterilerindir.")