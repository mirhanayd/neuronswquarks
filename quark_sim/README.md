# 🔬 Cornell Potential Simulation with Deep Inelastic Scattering
## Cornell Potansiyeli ve Derin İnelastik Saçılma Simülasyonu

A neural network-based quantum chromodynamics (QCD) simulation that models quark-antiquark interactions using the Cornell potential and visualizes Deep Inelastic Scattering (DIS) with electron trajectories.

Kuark-antikuark etkileşimlerini Cornell potansiyeli ile modelleyen ve Deep Inelastic Scattering (DIS) elektron yörüngelerini görselleştiren sinir ağı tabanlı kuantum kromodinamik (QCD) simülasyonu.

---

## 📋 Table of Contents / İçindekiler

- [Features / Özellikler](#features--özellikler)
- [Prerequisites / Ön Gereksinimler](#prerequisites--ön-gereksinimler)
- [Installation / Kurulum](#installation--kurulum)
- [Usage / Kullanım](#usage--kullanım)
- [Project Structure / Proje Yapısı](#project-structure--proje-yapısı)
- [Technical Details / Teknik Detaylar](#technical-details--teknik-detaylar)
- [Author / Geliştirici](#author--geliştirici)

---

## ✨ Features / Özellikler

### English
- **Cornell Potential Training**: 4-layer neural network learns the Cornell potential V(r) = -4αₛ/(3r) + kr
- **Deep Inelastic Scattering**: Simulates 20 electrons scattering off a quark target
- **Interactive GUI**: Real-time visualization with three panels:
  - Training loss convergence
  - Cornell potential comparison (Theory vs NN)
  - Electron trajectories in DIS simulation
- **Session Management**: Save/load simulation sessions as JSON
- **Organized Outputs**: Timestamped folders for each run
- **Bilingual**: Turkish and English output support

### Türkçe
- **Cornell Potansiyeli Eğitimi**: 4 katmanlı sinir ağı Cornell potansiyelini öğrenir V(r) = -4αₛ/(3r) + kr
- **Derin İnelastik Saçılma**: 20 elektronun kuark hedefine saçılmasını simüle eder
- **İnteraktif GUI**: Üç panelli gerçek zamanlı görselleştirme:
  - Eğitim kaybı yakınsaması
  - Cornell potansiyeli karşılaştırması (Teori vs NN)
  - DIS simülasyonunda elektron yörüngeleri
- **Oturum Yönetimi**: Simülasyon oturumlarını JSON olarak kaydet/yükle
- **Düzenli Çıktılar**: Her çalıştırma için zaman damgalı klasörler
- **İki Dilli**: Türkçe ve İngilizce çıktı desteği

---

## 🔧 Prerequisites / Ön Gereksinimler

### Step 1: Install Rust / Rust Kurulumu

#### Windows

1. **Download Rust Installer / Rust Yükleyiciyi İndirin**
   - Visit / Ziyaret edin: https://rustup.rs/
   - Download `rustup-init.exe` / İndirin: `rustup-init.exe`

2. **Run the Installer / Yükleyiciyi Çalıştırın**
   ```powershell
   # Double-click rustup-init.exe or run in PowerShell:
   # rustup-init.exe dosyasına çift tıklayın veya PowerShell'de çalıştırın:
   .\rustup-init.exe
   ```

3. **Follow the Installation / Kurulumu Takip Edin**
   - Press `1` for default installation / Varsayılan kurulum için `1` tuşlayın
   - Wait for completion / Tamamlanmasını bekleyin

4. **Verify Installation / Kurulumu Doğrulayın**
   ```powershell
   # Restart PowerShell, then check:
   # PowerShell'i yeniden başlatın, ardından kontrol edin:
   rustc --version
   cargo --version
   ```

#### Linux / macOS

```bash
# Install Rust / Rust Kur
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Restart terminal, then verify / Terminali yeniden başlat, sonra doğrula
rustc --version
cargo --version
```

### Step 2: Install Visual Studio Code / VS Code Kurulumu

1. **Download VS Code / VS Code İndir**
   - Visit / Ziyaret edin: https://code.visualstudio.com/
   - Download and install / İndirin ve kurun

2. **Install Rust Extension (Recommended) / Rust Eklentisi Kurun (Önerilen)**
   - Open VS Code / VS Code'u açın
   - Go to Extensions (`Ctrl+Shift+X`) / Eklentiler'e gidin (`Ctrl+Shift+X`)
   - Search "rust-analyzer" / "rust-analyzer" arayın
   - Click Install / Yükle'ye tıklayın

---

## 📥 Installation / Kurulum

### Clone the Repository / Depoyu Klonlayın

```powershell
# Windows PowerShell
cd C:\Users\YourUsername\Documents
git clone https://github.com/mirhanayd/neuronswquarks.git
cd neuronswquarks\quark_sim
```

```bash
# Linux / macOS
cd ~/Documents
git clone https://github.com/mirhanayd/neuronswquarks.git
cd neuronswquarks/quark_sim
```

### Open in VS Code / VS Code'da Açın

```powershell
# Windows
code .
```

```bash
# Linux / macOS
code .
```

Or manually / Veya manuel olarak:
1. Open VS Code / VS Code'u açın
2. `File > Open Folder` / `Dosya > Klasör Aç`
3. Navigate to `quark_sim` folder / `quark_sim` klasörüne gidin

---

## 🚀 Usage / Kullanım

### Option 1: Run New Simulation / Yeni Simülasyon Çalıştır

Open terminal in VS Code (`Ctrl+ö` or `Terminal > New Terminal`) and run:

VS Code'da terminal açın (`Ctrl+ö` veya `Terminal > New Terminal`) ve çalıştırın:

```powershell
# Build and run in release mode (optimized, ~2-3 minutes)
# Release modunda derle ve çalıştır (optimize, ~2-3 dakika)
cargo run --release
```

**What happens / Ne olur:**
1. ✅ Creates timestamped output folder: `outputs/YYYYMMDD_HHMMSS_GMT/`
2. ✅ Generates 3000 training samples
3. ✅ Trains 4-layer neural network for 8000 epochs
4. ✅ Tests Cornell potential predictions
5. ✅ Simulates 20 electrons in DIS
6. ✅ Saves 4 files:
   - `training_loss.svg` - Training loss plot
   - `cornell_potential.svg` - Potential comparison
   - `scattering.svg` - Electron trajectories
   - `session.json` - Session data
7. ✅ Opens interactive GUI window

**Output Example / Çıktı Örneği:**
```
🚀 Cornell Potansiyeli Simülasyonu / Cornell Potential Simulation
📁 Çıktı klasörü / Output folder: outputs/20251203_143052_GMT

🖥️  Cihaz / Device: CPU
📊 Eğitim verisi oluşturuluyor / Generating training data...
   ✓ 3000 veri noktası oluşturuldu / data points generated

🎓 Eğitim başlıyor / Starting training...
   • Epoch sayısı / Epochs: 8000
   • Öğrenme oranı / Learning rate: 0.0200

Epoch 0: Kayıp / Loss (MSE) = 12.084 GeV²
Epoch 7500: Kayıp / Loss (MSE) = 0.016 GeV²

✅ Eğitim tamamlandı / Training completed!

⚛️ Deep Inelastic Scattering Simülasyonu / DIS Simulation
   ✓ Saçılma simülasyonu tamamlandı
   📊 Geniş açı (>10°): 18 elektron
       Küçük açı (<10°): 2 elektron

💾 Oturum kaydediliyor / Saving session...
   ✓ outputs/20251203_143052_GMT/session.json
```

---

### Option 2: Load Trained Model (⚡ FASTEST - 1 Second!) / Eğitilmiş Model Yükle (⚡ EN HIZLI - 1 Saniye!)

**🧠 The Smart Way / Akıllı Yöntem:**

If you already trained once, you can skip the 2-3 minute training and load the pre-trained "brain":

Bir kez eğitim yaptıysanız, 2-3 dakikalık eğitimi atlayıp önceden eğitilmiş "beyni" yükleyebilirsiniz:

```powershell
# Load trained model weights (⚡ 1 second startup!)
# Eğitilmiş model ağırlıklarını yükle (⚡ 1 saniye başlangıç!)
cargo run --release -- --load-model outputs/20251203_143052_GMT/trained_model.safetensors
```

**What happens / Ne olur:**
1. ✅ Loads neural network weights from `.safetensors` file
2. ✅ Skips 5000 epochs of training (saves 2-3 minutes!)
3. ✅ Runs DIS simulation with pre-trained model
4. ✅ Opens GUI immediately
5. ✅ **Total time: ~5 seconds** ⚡

**When to use / Ne zaman kullanılır:**
- After first training / İlk eğitimden sonra
- Testing different DIS parameters / Farklı DIS parametrelerini test ederken
- Quick demonstrations / Hızlı gösterimler için
- Production deployments / Üretim ortamları için

**Format:** SafeTensors (Hugging Face standard, used by Stable Diffusion, BERT, etc.)

---

### Option 3: Load Previous Session / Önceki Oturumu Yükle

If you want to reload exact previous results (including plots):

Tam önceki sonuçları (grafikler dahil) yüklemek istiyorsanız:

```powershell
# Load specific session / Belirli oturumu yükle
cargo run --release -- --load outputs/20251203_143052_GMT/session.json
```

**What happens / Ne olur:**
1. ✅ Loads saved neural network predictions
2. ✅ Loads electron trajectories
3. ✅ Opens GUI immediately (no training, no simulation)
4. ✅ ~3 seconds total

**When to use / Ne zaman kullanılır:**
- Want to review previous results / Önceki sonuçları gözden geçirmek istiyorsanız
- Need to compare different runs / Farklı çalıştırmaları karşılaştırmak istiyorsanız
- Presenting results quickly / Sonuçları hızlıca sunmak istiyorsanız

---

### Alternative: Quick Debug Build / Alternatif: Hızlı Debug Derleme

```powershell
# Faster compilation, slower execution (~5-6 minutes for training)
# Daha hızlı derleme, daha yavaş çalışma (~5-6 dakika eğitim)
cargo run
```

---

## 📂 Project Structure / Proje Yapısı

```
quark_sim/
├── src/
│   ├── main.rs          # Main orchestration / Ana orkestrasyon
│   ├── physics.rs       # Cornell potential formula / Cornell potansiyel formülü
│   ├── model.rs         # Neural network architecture / Sinir ağı mimarisi
│   ├── training.rs      # Training & testing logic / Eğitim & test mantığı
│   ├── plotting.rs      # SVG & terminal plots / SVG & terminal grafikleri
│   ├── gui.rs           # Interactive GUI with egui / Etkileşimli GUI
│   └── scattering.rs    # DIS simulation / DIS simülasyonu
├── outputs/             # Timestamped result folders / Zaman damgalı sonuç klasörleri
│   └── YYYYMMDD_HHMMSS_GMT/
│       ├── training_loss.svg           # Training convergence plot
│       ├── cornell_potential.svg       # Potential comparison
│       ├── scattering.svg              # DIS electron trajectories
│       ├── session.json                # Complete session data
│       └── trained_model.safetensors   # 🧠 Neural network weights (reusable!)
├── Cargo.toml           # Rust dependencies / Rust bağımlılıkları
└── README.md
```

---

## 🔬 Technical Details / Teknik Detaylar

### Physics / Fizik

**Cornell Potential / Cornell Potansiyeli:**
```
V(r) = -4αₛ/(3r) + kr

where / burada:
- αₛ = 0.5 (strong coupling constant / güçlü etkileşim sabiti)
- k = 0.9 GeV/fm (string tension / sicim gerilimi)
- r: quark-antiquark distance / kuark-antikuark mesafesi
```

**Deep Inelastic Scattering:**
- 20 electrons fired at quark target / 20 elektron kuark hedefine fırlatılır
- Force calculation: F = -∇V / Kuvvet hesabı: F = -∇V
- Trajectory integration with dt = 0.05 fm/c
- Impact parameter range: ±2.0 fm

### Neural Network / Sinir Ağı

```
Architecture / Mimari: 3 → 128 → 64 → 32 → 1
Activation / Aktivasyon: ReLU
Optimizer / Optimize edici: SGD (Stochastic Gradient Descent)
Learning Rate / Öğrenme Oranı: 0.02
Epochs / Dönem: 8000
Training Samples / Eğitim Örnekleri: 3000
```

### Dependencies / Bağımlılıklar

- **candle-core** (0.9.1): ML framework / ML çerçevesi
- **eframe** (0.29): GUI framework / GUI çerçevesi
- **plotters** (0.3): SVG plotting / SVG grafik
- **serde** (1.0): JSON serialization / JSON serileştirme
- **chrono** (0.4): Timestamps / Zaman damgaları

---

## 🎨 GUI Features / GUI Özellikleri

The interactive window shows / İnteraktif pencere gösterir:

1. **Training Loss Panel / Eğitim Kaybı Paneli**
   - MSE loss over epochs / Epoch'lara göre MSE kaybı
   - Toggle visibility / Görünürlüğü aç/kapa

2. **Cornell Potential Panel / Cornell Potansiyel Paneli**
   - Theoretical curve (blue) / Teorik eğri (mavi)
   - Neural network prediction (red) / Sinir ağı tahmini (kırmızı)
   - Test points (green) / Test noktaları (yeşil)

3. **DIS Scattering Panel / DIS Saçılma Paneli**
   - 20 electron trajectories (colored) / 20 elektron yörüngesi (renkli)
   - Quark target (red dot) / Kuark hedefi (kırmızı nokta)
   - Statistics: large/small angle scattering / İstatistikler: geniş/küçük açı saçılma

---

## 📊 Expected Results / Beklenen Sonuçlar

### Training / Eğitim
- Initial loss / İlk kayıp: ~9-12 GeV²
- Final loss / Son kayıp: ~0.01-0.03 GeV²
- Convergence time / Yakınsama süresi: ~2-3 minutes (release mode)

### Cornell Potential Accuracy / Cornell Potansiyel Doğruluğu
- Mid-range distances (0.2-2.0 fm): 2-7% error
- Extreme distances: Higher error (expected)

### DIS Statistics / DIS İstatistikleri
- Large angle (>10°): ~15-18 electrons
- Small angle (<10°): ~2-5 electrons

---

## 🛠️ Troubleshooting / Sorun Giderme

### Compilation Errors / Derleme Hataları

**Problem:** `rustc` not found
**Solution / Çözüm:**
```powershell
# Restart terminal after Rust installation
# Rust kurulumundan sonra terminali yeniden başlat
rustup update
```

**Problem:** Linker errors on Windows
**Solution / Çözüm:**
- Install Visual Studio C++ Build Tools
- https://visualstudio.microsoft.com/downloads/

### Runtime Issues / Çalışma Zamanı Sorunları

**Problem:** GUI doesn't open
**Solution / Çözüm:**
- Check graphics drivers / Grafik sürücülerini kontrol edin
- Try debug mode: `cargo run` / Debug modu deneyin: `cargo run`

**Problem:** Training takes too long
**Solution / Çözüm:**
- Use release mode: `cargo run --release`
- Release modu kullanın: `cargo run --release`

---

## 📝 Example Commands / Örnek Komutlar

```powershell
# Full workflow / Tam iş akışı
cargo run --release                                          # New simulation with training / Eğitimli yeni simülasyon (~2-3 min)
cargo run --release -- --load-model outputs/20251203_143052_GMT/trained_model.safetensors  # ⚡ Load trained brain (1 sec)
cargo run --release -- --load outputs/20251203_143052_GMT/session.json  # Load exact results / Tam sonuçları yükle (3 sec)

# Development / Geliştirme
cargo build                                                  # Build only / Sadece derle
cargo check                                                  # Quick syntax check / Hızlı sözdizimi kontrolü
cargo clean                                                  # Clean build artifacts / Derleme dosyalarını temizle
cargo fmt                                                    # Format code / Kodu biçimlendir
```

---

## 🌟 Contributing / Katkıda Bulunma

Contributions are welcome! / Katkılar hoş karşılanır!

1. Fork the repository / Depoyu fork edin
2. Create feature branch / Özellik dalı oluşturun: `git checkout -b feature-name`
3. Commit changes / Değişiklikleri commit edin: `git commit -m "Add feature"`
4. Push to branch / Dala push edin: `git push origin feature-name`
5. Open Pull Request / Pull Request açın

---

## 📄 License / Lisans

This project is open source and available for educational purposes.

Bu proje açık kaynaklıdır ve eğitim amaçlı kullanıma açıktır.

---

## 👨‍💻 Author / Geliştirici

**Mirhan Aydın**

GitHub: [@mirhanayd](https://github.com/mirhanayd)

---

## 🙏 Acknowledgments / Teşekkürler

- Cornell potential model from QCD theory / QCD teorisinden Cornell potansiyel modeli
- Candle ML framework by Hugging Face / Hugging Face'ten Candle ML çerçevesi
- Deep Inelastic Scattering physics / Derin İnelastik Saçılma fiziği

---

**⚡ Quick Start / Hızlı Başlangıç:**
```powershell
# First time (with training) / İlk kez (eğitimle)
git clone https://github.com/mirhanayd/neuronswquarks.git
cd neuronswquarks/quark_sim
cargo run --release

# Next times (skip training) / Sonraki seferler (eğitimi atla)
cargo run --release -- --load-model outputs/LATEST_FOLDER/trained_model.safetensors
```

**Pro Tip / İpucu:** After first run, always use `--load-model` to save time! The trained "brain" is reusable.

**Enjoy simulating quantum physics! / Kuantum fiziği simülasyonunun tadını çıkarın! 🔬✨**
