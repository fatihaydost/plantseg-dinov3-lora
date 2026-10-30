Original Turkish pre-registration, committed 2026-10-10 before the first run; English translation in preregistration.md.

# T1 PlantSeg × DINOv3: ön kayıt (koşmadan önce yazıldı, 10.10.2026 ~21:40)

Bu dosya ilk koşudan önce yazıldı ve sonra değiştirilmez. Sapmalar `sapmalar.md`'de.

## Veri ve split
- PlantSeg (Wei vd., Sci Data 2026, DOI 10.1038/s41597-025-06513-4; Zenodo 17719108, CC BY-NC 4.0).
- Resmi 70/10/20 split dosyası (Metadata.csv). **Eğitim = resmi train, dev = resmi validation (%10).** Test (%20)
  hiçbir koşuda yüklenmez, hiçbir seçim ona bakılarak yapılmaz.
- Veri hijyeni (model yok, etiket yok): pHash ile train↔val ve train↔test yakın kopya sayımı (yalnız görüntü hash'i;
  kartın K/sızıntı maddesi bunu istiyor). Bu sayım hiçbir model veya ayar seçimine girmez.
- Sınıf: 116 (arka plan + 115 hastalık), veri makalesi kodundaki gibi `num_classes=116`, mIoU 116 sınıf üzerinden
  (mmseg IoUMetric: dev'de hiç görünmeyen sınıf ortalamaya girmez, nan atlanır).

## Ölçüt
- Birincil: dev mIoU (%). İkincil: mAcc. Değerlendirme: kısa kenar 512'ye yeniden boyut (veri makalesi
  `Resize scale=(2048,512) keep_ratio`), tüm görüntü çıkarımı, logit orijinal etiket çözünürlüğüne bilineer büyütülür.
  TTA yok.

## Karşılaştırılacaklar (hepsi aynı ön işlem, aynı başlık, aynı tekrar sayısı, aynı veri)
- **Başlık (hepsinde aynı):** UperNet-lite = PPM + FPN, 256 kanal, 4 seviye girdi (1/4, 1/8, 1/16, 1/32).
  ViT'ler için 4 ara katman (ViT-B: blok 3, 6, 9, 12) + basit özellik piramidi (ViTDet tarzı: 2× deconv, 1×,
  identity, maxpool) başlığın parçası sayılır.
- **B0 taban (küçük ölçekli yeniden üretim):** ConvNeXt-T, ImageNet-22k ön eğitimli (timm `convnext_tiny.fb_in22k`),
  tam ince ayar. Veri makalesinin ConvNeXt-L UperNet'inin (46,24 mIoU test, 40k iter × 16 batch, 512 crop) 4 GB GPU'ya
  sığan en yakın hâli. Makale sayısı (46,24 / ConvNeXt-B 40,69 / Swin-B 38,67) ayrıca yazılır, B0 ile doğrudan
  karşılaştırılmaz (farklı split: dev vs test, farklı boyut, farklı takvim).
- **P1 öneri, en basit (H1 birincil):** DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3.lvd1689m`), **donmuş** + aynı başlık.
- **P2 öneri, uyarlama (H1 ikincil):** DINOv3 ViT-B/16 + LoRA (r=16, qkv ve proj) + aynı başlık. Zaman kalırsa.
- **M2 güçlü modern ek taban:** DINOv2 ViT-B/14 donmuş + aynı başlık (v3 mi, genel FM mi sorusu için). Zaman kalırsa.
- **H2 en basit:** P2 (ya da P2 koşmazsa P1) + kuyruk-duyarlı kayıp (logit ayarı, τ=1, eğitim piksel frekansından)
  vs aynı model düz CE. Zaman kalırsa; yalnız sinyal.

## Eğitim (hepsinde aynı)
- Artırma: veri makalesi hattı (RandomResize scale (2048,512) oran 0,5-2,0, RandomCrop 512 cat_max_ratio 0,75,
  flip, fotometrik bozulma). 4 GB'a sığmazsa crop hepsinde birlikte küçültülür ve sapmaya yazılır.
- AdamW, lr 2e-4 (başlık ve LoRA), ince ayarlı omurga 2e-4 × 0,1, wd 0,01,
  poly takvim + 300 iter ısınma, bf16/fp16 karışık hassasiyet, efektif batch 8 (gradyan biriktirme gerekirse).
- **Takvim:** her koşu aynı iterasyon sayısı N (aynı görülen örnek). N, throughput ölçülünce bir koşu ≤ ~75 dk
  olacak şekilde seçilir ve ilk koşudan önce `sapmalar.md`'ye değil, `log/plan.txt`'ye yazılır; tüm koşular aynı N.
- Seçim: son iterasyondaki model (erken durdurma yok, dev'e bakarak checkpoint seçimi yok). Dev yalnız sonda ölçülür.

## Tohum
- B0 ve P1: 2 tohum (0, 1). P2/M2/H2: zaman kalırsa 2, olmazsa 1 (tek tohum = yalnız sinyal diye yazılır).

## Hipotezler ve karar kuralı
- **H1 (güvenli):** P1 (donmuş DINOv3 ViT-B + aynı başlık), aynı protokoldeki B0'ı dev mIoU'da **δ = +2,0 puan**
  geçer (2 tohum ortalaması farkı ≥ 2,0 ve P1'in her tohumu B0'ın her tohumundan yüksek).
  - δ gerekçesi: kartın δ'sı 2,0; Q2 çıpalarının δ medyanı ~+1,4'ün üstünde (güvenli tarafta); ~1.100 görüntülük dev'de
    tohum gürültüsü segmentasyonda tipik 0,3-1 puan, 2 puan bunun açıkça üstünde.
  - P1 tutmaz ama P2 tutarsa: "H1 donmuş biçimde tutmadı, LoRA ile tuttu" diye yazılır (daha zayıf sonuç).
- **H2 (iddialı, en basit hâli):** kuyruk-duyarlı kayıp, düz CE'yi dev mIoU'da ≥ +1,5 geçer (kart δ₂). Bu duman testinde
  yalnız sinyal (tohum azsa karar verilmez).
- **H3 (analiz):** P1 − B0 farkı sınıf frekansı üçte birlerinde (baş/orta/kuyruk, eğitim görüntü sayısına göre) ve
  maske oranı (küçük lezyon: görüntüde maske oranı < medyan) alt kümelerinde ayrı raporlanır.

## Süre bütçesi
- Tek koşu ≤ 3 sa (hedef ≤ 75 dk), toplam ≤ 1 gün; GPU başka deneyciyle `flock` ile paylaşılıyor.

## Bilinen sınırlılıklar (önceden)
- Tam ConvNeXt-L/40k yeniden üretimi 4 GB'da yapılmayacak; E1'in "±2 puan" şartı ancak küçük ölçekli taban ve makale
  sayısı yan yana yazılarak tartışılır.
- Veri makalesi kodu dev yerine test'i doğrulama olarak kullanıyor (`val_dataloader` → `images/test`); bizim protokol
  dev'i kullanır, bu yüzden makale sayısı ile mutlak kıyas yanlı olabilir.
