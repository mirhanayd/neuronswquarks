// Cornell Potansiyeli Sinir Ağı Simülasyonu
// Kuark-antikuark etkileşimlerini modelleyen derin öğrenme projesi

mod physics;     // Fizik sabitleri ve Cornell potansiyeli
mod model;       // Sinir ağı modeli
mod training;    // Eğitim ve test fonksiyonları
mod plotting;    // Grafik çizim
mod gui;         // GUI bileşenleri
mod scattering;  // Deep Inelastic Scattering simülasyonu

use candle_core::{Device, Result};
use chrono::Utc;

use gui::{AppData, launch_gui};
use plotting::{plot_results, show_terminal_plots};
use training::{generate_training_data, train_model, test_model, create_model_and_optimizer};
use scattering::{simulate_scattering, plot_scattering, ScatteringParams};

fn main() -> Result<()> {
    // Komut satırı argümanlarını kontrol et
    let args: Vec<String> = std::env::args().collect();
    
    // Eğer --load parametresi varsa, kayıtlı oturumu yükle
    if args.len() >= 3 && args[1] == "--load" {
        let session_file = &args[2];
        println!("📂 Kayıtlı oturum yükleniyor / Loading saved session: {}", session_file);
        
        match AppData::load_session(session_file) {
            Ok(app_data) => {
                println!("✅ Oturum başarıyla yüklendi / Session loaded successfully!");
                println!("🖥️  İnteraktif GUI penceresi açılıyor / Opening interactive GUI...");
                launch_gui(app_data, "Cornell Potansiyeli - Kayıtlı Oturum");
                return Ok(());
            }
            Err(e) => {
                eprintln!("❌ Oturum yüklenemedi / Session load failed: {}", e);
                eprintln!("Yeni simülasyon başlatılıyor / Starting new simulation...\n");
            }
        }
    }
    
    // Normal akış: Yeni simülasyon çalıştır
    println!("🚀 Cornell Potansiyeli Simülasyonu / Cornell Potential Simulation");
    println!("   Kuark-Antikuark Etkileşim Modeli / Quark-Antiquark Interaction Model\n");
    
    // 0. Zaman damgası ve çıktı klasörü oluştur
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S_GMT").to_string();
    let output_dir = format!("outputs/{}", timestamp);
    std::fs::create_dir_all(&output_dir)?;
    println!("📁 Çıktı klasörü / Output folder: {}\n", output_dir);
    
    // 1. Cihaz Seçimi
    let device = Device::Cpu;
    println!("🖥️  Cihaz / Device: CPU");
    
    // 2. Eğitim Verisi Oluştur
    println!("\n📊 Eğitim verisi oluşturuluyor / Generating training data...");
    let n_samples = 3000;
    let (distances, target, target_mean, target_std) = generate_training_data(n_samples, &device)?;
    println!("   ✓ {} veri noktası oluşturuldu / data points generated", n_samples);
    
    // 3. Model ve Optimizer Oluştur
    println!("\n🧠 Model oluşturuluyor / Creating model...");
    let (model, mut optimizer, _varmap) = create_model_and_optimizer(&device, 0.02)?;
    println!("   ✓ 4 katmanlı sinir ağı / 4-layer neural network (3→128→64→32→1)");
    
    // 4. Eğitim
    let epochs = 8000;
    let loss_history = train_model(
        &model,
        &mut optimizer,
        &distances,
        &target,
        target_mean,
        target_std,
        epochs,
        &device,
    )?;
    
    // 5. Test
    let (test_distances, cornell_values, nn_values, potential_points_theory, potential_points_nn) = 
        test_model(&model, target_mean, target_std, &device)?;
    
    // 6. Grafikleri Oluştur
    println!("\n📈 Grafikler oluşturuluyor / Generating plots...");
    let (loss_file, potential_file) = plot_results(
        &output_dir,
        &loss_history,
        &test_distances,
        &cornell_values,
        &nn_values,
        &model,
        target_mean,
        target_std,
        &device,
    );
    
    // 7. Terminal Grafikleri Göster
    show_terminal_plots(&loss_history, &test_distances, &cornell_values, &nn_values);
    
    // 7.5. Deep Inelastic Scattering Simülasyonu
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎯 DEEP INELASTIC SCATTERING SIMÜLASYONU BAŞLIYOR");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let scattering_params = ScatteringParams::default();
    let electrons = simulate_scattering(
        &model,
        &scattering_params,
        target_mean,
        target_std,
        &device,
    )?;
    
    // DIS grafiğini kaydet
    let scattering_file = format!("{}/scattering.svg", output_dir);
    plot_scattering(&electrons, &scattering_file);
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // 8. Oturum Verilerini Hazırla
    // Elektron verilerini ElectronData formatına dönüştür
    let electron_data: Vec<gui::ElectronData> = electrons.iter().map(|e| {
        gui::ElectronData {
            trajectory: e.trajectory.clone(),
            impact_parameter: e.impact_parameter,
        }
    }).collect();
    
    let app_data = AppData {
        loss_history: loss_history.clone(),
        potential_theory: potential_points_theory,
        potential_nn: potential_points_nn,
        test_distances: test_distances.clone(),
        cornell_values: cornell_values.clone(),
        nn_values: nn_values.clone(),
        loss_file: loss_file.clone(),
        potential_file: potential_file.clone(),
        scattering_file: Some(scattering_file.clone()),
        electrons: Some(electron_data),
    };
    
    // 9. Oturumu Kaydet
    println!("\n💾 Oturum kaydediliyor / Saving session...");
    match app_data.save_session(&output_dir) {
        Ok(session_file) => {
            println!("   ✓ {}", session_file);
            println!("\n📝 Daha sonra açmak için / To load later:");
            println!("   cargo run --release -- --load {}", session_file);
        }
        Err(e) => {
            eprintln!("   ⚠️  Oturum kaydedilemedi / Session save failed: {}", e);
        }
    }
    
    // 10. GUI Başlat
    println!("\n✅ Simülasyon tamamlandı / Simulation completed!");
    println!("🖥️  İnteraktif GUI penceresi açılıyor / Opening interactive GUI...\n");
    launch_gui(app_data, "Cornell Potansiyeli - Sinir Ağı Simülasyonu");

    Ok(())
}
