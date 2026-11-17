# Projeyi anlama ve savunma rehberi

## Ne geliştirdik?

50x50 hücrede üç bağımsız ajanın yiyecek topladığı bir ortam. Yeşil +1, kırmızı -1.
Üç karar verme yöntemi var: rastgele seçim, kurallı sezgisel seçim ve deneyimden
öğrenen Double DQN. Başarı iddiası results/summary.json içindeki deneylerle sınırlı.

## Bir adımda ne oluyor?

1. Her ajan kendi gözlem vektörünü alır. Vektör bütün yiyecekleri, kendi konumunu ve
   yalnızca en yakın komşusunun konumunu içerir.
2. Politika bu vektörü kullanarak beş eylemden birini seçer.
3. Ortam üç eylemi birlikte işler; hızlar nedeniyle hareketi alt adımlara ayırır.
4. Aynı yiyeceğe talepler çözülür. Yeşil rekabetinde sola hareket eden önceliklidir.
5. Yiyecek silinir, yalnızca tüketenin sayacı ve ödülü güncellenir.
6. Yeni hızlar ve gözlemler hazırlanır. Eğitimde bu geçiş deneyim tamponuna eklenir.

## Önemli kodların görevi

Environment.step eylemleri doğrular ve tek ortak geçiş üretir. MovementResolver
önce bütün hareketleri hesaplar, sonra yiyecek taleplerini birlikte çözer; böylece
listede ilk duran ajan otomatik avantaj kazanmaz. ObservationManager bilgi
erişiminin sınırıdır. RewardManager sadece +1/-1 kuralını uygular.

HeuristicAgent görünür yeşiller arasından uzaklık ve komşu rekabetine göre hedef
seçer. Olası hareketleri yol üstü yiyecekler ve kırmızı riskini içeren Dijkstra yol maliyetine göre puanlar. Bekleme maliyeti karar puanına aittir; çevre ödülü değildir.
Bu puan öğrenilmiş bir sonuç değildir; geliştiricinin yazdığı karar kuralıdır.

## RL gerçekten ne öğreniyor?

Q(o,a), o gözleminde a eylemi seçilirse beklenen indirimli gelecek ödülü tahminidir.
Her ajanın ayrı 22-32-1 sinir ağı bulunur. Aynı ağ beş aday eylemi ayrı değerlendirir.
Ağ girişleri görünür yiyeceklerin hareket yolundaki varlığı, hedef geometrisi, hız
ve kalan süre gibi özelliklerden türetilir. Eylemi belirleyen sayısal ağırlıklar
deneyimlerden öğrenilir. Heuristic kararı RL'ye kopyalanmaz, öğretmen verisi yoktur.

Ham gözlem 2508 boyutludur. Öğrenme için her eylem 22 özellik ile temsil edilir.
Bu özellik mühendisliği öğrenmeyi kolaylaştırır ama tam haritayı sıkıştırırken
bilgi kaybettirir. Dolayısıyla bunu ham görüntüden uçtan uca öğrenme diye sunma.

İlk altı özellik, hareketin üç alt hücresindeki yeşil/kırmızı göstergeleridir.
Sonrakiler yakın yeşil yönü ve uzaklığı, gözlenen komşunun geometrisi, ilerleme,
hız, metrik, eylem kimliği, kalan yeşil oranı ve kalan süredir. Komşunun hedefe
uzaklığı özelliği Manhattan yaklaşımıdır; gözlemdeki komşu seçimi episode metriğiyle
yapılır. Bu fark bir gizli bilgi kullanımı değildir, yaklaşık bir tasarım tercihidir.

Epsilon-greedy bazen rastgele eylemle keşfeder, diğer zamanlarda en yüksek Q'yu
kullanır. Replay buffer geçmiş geçişlerden karışık mini-batch örnekler. Target
network daha yavaş güncellenen hedefler sağlar. Double DQN'de sonraki eylemi online
ağ seçer, o eylemin değerini target ağı hesaplar. gamma=.9 geleceği indirimler.

Model dosyaları çıkarım için ağırlıkları saklar. Son model de ayrıca kaydedilir;
tam eğitim sürdürme için optimizer/replay/RNG durumları saklanmaz. Eğitim komutu
başlangıçtan tekrarlanabilir; yükleme komutu kaydedilmiş politikayı çalıştırır.

## Üç ajan bağımsızsa birbirini nasıl etkiliyor?

Aynı yeşil kaynakları tüketirler. Birinin aldığı yiyecek diğerine kalmaz. Her ajan
kendi ağı ve bireysel ödülüyle öğrenir. Takım ödülü yalnızca raporlama için toplanır.
Aralarında mesajlaşma ya da bütün konumları bilen ortak karar verici yoktur.

## Deney sonuçlarını nasıl yorumlamalısın?

Eğitim, doğrulama ve test haritaları ayrıdır. En iyi model doğrulama haritalarıyla
seçilir. Son test haritalarında parametre ayarı yapılmaz. Tek iyi episode başarı
kanıtı değildir. Eğitim seed'leri ve haritalar arasındaki değişkenlik birlikte
raporlanır; aynı episode'un üç ajanı bağımsız üç deney gibi sayılmaz.

Grafiklerin gölgeli eğitim bandı +/-1 standart sapmadır; algoritma karşılaştırma
grafiğindeki hata çubukları %95 bootstrap güven aralığıdır. Bunları karıştırma.
Güven aralığı mutlak başarı garantisi değil, bu deney tasarımındaki belirsizliktir.

## Sunumda canlı gösterim sırası

1. Karma modda aynı haritada Random, Heuristic ve Double DQN'yi göster.
2. Duraklat, yeşil/kırmızı sayaçlarını ve bireysel toplamları açıkla.
3. Heuristic hedef işaretini ve hareket izlerini göster. RL sabit bir hedef seçmek
   yerine eylem değerleri seçer; arayüzde bunu açıkça belirtir.
4. Aynı seed ile sıfırla. Yeni harita ile görülmemiş başlangıcı göster.
5. Model yükleme düğmesiyle başka eğitim tekrarının modeline geç.
6. Testleri, öğrenme eğrisini, test sonuçlarını ve sınırlılıkları aç.

## Sık sorular

Neden NumPy? Küçük ağda forward, Huber türevi ve Adam açıkça denetlenebilir. Büyük
GPU eğitimi yapılmadığı için PyTorch zorunlu tutulmadı. Bu tercih daha büyük ağlara
ölçeklenme avantajı sunmaz.

Neden tüm oyun bir MDP değil? Tam ortam durumu geçişi belirleyebilir, ancak bir
ajan üçüncü ajanın konumunu görmez. Ajanın gözlemi kısmi ve özellik kodlaması da
kayıplıdır. DQN için genel yakınsama veya optimumluk garantisi ileri sürmüyoruz.

Neden adım cezası yok? Dersin ödülünü değiştirmemek için. Bekleme davranışı olursa
bunu ölçer ve raporlarız. Sezgisel puanlar çevre ödülü değildir.

Neden 300 adım deney, arayüzde 500? İkisi de ayarlanabilir sonlu ufuktur. Teslimdeki
modeller ve ana karşılaştırmalar 300 adımda üretildi. 500 adım ayrı koşuldur; 300
adımın sonuçlarını 500 adımlık kanıt diye sunmuyoruz.

## Teslimden önce kişisel kontrol

Öğretim üyesinin özgün yönergesiyle varsayımları karşılaştır. Kodun açıklamasını
öğren, gerekirse rapor
kapağına gerçek grup üyelerini ekle. Notun öğretim üyesince belirlendiğini unutma.
