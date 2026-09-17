# Birliktelik Kuralı Tabanlı Öneri Sistemi

Online Retail II veri seti üzerinde Apriori algoritmasıyla (Association Rule Learning) geliştirilen bir sepet bazlı öneri sistemi. [Miuul](https://miuul.com) Data Scientist Bootcamp kapsamında hazırlanmıştır.

## İş Problemi

Üç farklı müşterinin sepetinde birer ürün bulunuyor. **2010-2011 dönemi Alman müşteri işlemlerinden** türetilen birliktelik kurallarını kullanarak her sepeti tamamlayacak en uygun ürün(ler)i öner.

| Kullanıcı | Sepetteki Ürün (StockCode) |
|---|---|
| Kullanıcı 1 | 21987 |
| Kullanıcı 2 | 23235 |
| Kullanıcı 3 | 22747 |

## Veri Seti

**Online Retail II** — İngiltere merkezli bir hediyelik eşya perakendecisinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içerir (müşterilerin çoğu toptancıdır).

| Değişken | Açıklama |
|---|---|
| `Invoice` | Fatura numarası ("C" ile başlıyorsa iptal edilmiş demektir) |
| `StockCode` | Eşsiz ürün kodu |
| `Description` | Ürün adı |
| `Quantity` | Satılan adet |
| `InvoiceDate` | Fatura tarihi |
| `Price` | Birim fiyat (Sterlin) |
| `Customer ID` | Eşsiz müşteri numarası |
| `Country` | Müşterinin ülkesi |

Kaynak: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (kullanılan sayfa: `Year 2010-2011`). Ham dosya (~45 MB) bu repoya dahil edilmemiştir — indirme talimatları için [`data/README.md`](data/README.md) dosyasına bakınız.

## Yaklaşım

**1. Veri Hazırlama**
- `StockCode == "POST"` olan satırları çıkar (kargo bedeli, ürün değil)
- Eksik değerleri çıkar
- İptal edilen faturaları çıkar (`Invoice` içinde "C" geçenler)
- Sadece `Price > 0` olan satırları tut
- `Price` ve `Quantity` değişkenlerindeki aykırı değerleri IQR yöntemiyle baskıla

**2. Birliktelik Kuralı Madenciliği (Alman Müşteriler)**
- Fatura × ürün sepet matrisini oluştur (`create_invoice_product_df`)
- `mlxtend` ile Apriori çalıştır ve kuralları türet (`create_rules`)

**3. Öneri**
- Her sepet ürünü için kuralları **lift** değerine göre sırala
- En güçlü eşleşen kural(lar)ın sonuç (consequent) ürün(ler)ini öner (`arl_recommender`)

## Sonuçlar

| Kullanıcı | Sepetteki ürün | Önerilen |
|---|---|---|
| Kullanıcı 1 | 21987 — PACK OF 6 SKULL PAPER CUPS | 21086 — SET/6 RED SPOTTY PAPER CUPS, 21988 — PACK OF 6 SKULL PAPER PLATES |
| Kullanıcı 2 | 23235 — STORAGE TIN VINTAGE LEAF | 23243 — SET OF TEA COFFEE SUGAR TINS PANTRY, 23244 — ROUND STORAGE TIN VINTAGE LEAF |
| Kullanıcı 3 | 22747 — POPPY'S PLAYHOUSE BATHROOM | 22746 — POPPY'S PLAYHOUSE LIVINGROOM, 22745 — POPPY'S PLAYHOUSE BEDROOM |

Tam konsol çıktısı [`outputs/recommendation_results.txt`](outputs/recommendation_results.txt) dosyasında kayıtlıdır.

## Proje Yapısı

```
.
├── data/
│   └── README.md              # online_retail_II.xlsx nereden indirilir
├── outputs/
│   └── recommendation_results.txt
├── src/
│   └── arl_recommender.py     # tam çözüm
├── requirements.txt
└── README.md
```

## Çalıştırma

```bash
pip install -r requirements.txt
# online_retail_II.xlsx dosyasını data/ klasörüne yerleştirin (bkz. data/README.md)
python src/arl_recommender.py
```

## Teknik Notlar

- Kod **pandas 3.0** ile test edilmiştir; bu sürüm, aykırı değer eşiği gibi bir float değeri `int64` tipindeki bir sütuna (örn. `Quantity`) yazmaya çalışırken `LossySetitemError` fırlatır — sütun baskılamadan önce `float64`'e dönüştürülür.
- `mlxtend`'in Apriori fonksiyonu boolean bir sepet matrisi bekler — pivot tablo `.astype(bool)` ile dönüştürülür.

## Kullanılan Araçlar

`pandas` · `mlxtend` (Apriori / Association Rules) · `openpyxl`

---
*Miuul Data Scientist Bootcamp vaka çalışmaları serisinin bir parçasıdır.*
