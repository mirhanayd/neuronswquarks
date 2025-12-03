mod physics;
mod model;
mod training;
mod plotting;
mod gui;
mod scattering;

use candle_core::{Device, Result};
use chrono::Utc;
use std::path::Path;
use std::sync::Arc; // Arc ekledik

use gui::{AppData, launch_gui, InteractiveContext};
use plotting::plot_results;
use training::{generate_training_data, train_model, test_model, create_model_and_optimizer, save_model_with_config, load_model_with_config};
use scattering::{simulate_scattering, plot_scattering, ScatteringParams, get_proton_quarks};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // --load (Sadece Kayıt İzleme - Canlı Mod Yok)
    if args.len() >= 3 && args[1] == "--load" {
        let session_file = &args[2];
        match AppData::load_session(session_file) {
            Ok(app_data) => {
                // İnteraktif context yok (None), çünkü model yok
                launch_gui(app_data, "Kayıtlı Oturum (İzleme Modu)", None);
                return Ok(());
            }
            Err(e) => eprintln!("Hata: {}", e),
        }
    }
    
    // --load-model (Eğitilmiş Modeli Yükle - CANLI MOD AKTİF!)
    if args.len() >= 3 && args[1] == "--load-model" {
        let model_path = &args[2];
        let config_path = model_path.replace(".safetensors", "_config.json");
        
        if Path::new(&config_path).exists() {
            return run_with_pretrained_model(model_path, &config_path);
        } else {
            println!("⚠️ UYARI: Config dosyası bulunamadı.");
            return Ok(());
        }
    }
    
    // --- NORMAL EĞİTİM VE BAŞLATMA ---
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S_GMT").to_string();
    let output_dir = format!("outputs/{}", timestamp);
    std::fs::create_dir_all(&output_dir).expect("Klasör oluşturulamadı");
    println!("📁 Çıktılar: {}", output_dir);
    
    let device = Device::Cpu;
    println!("\n📊 Veri seti hazırlanıyor...");
    let (distances, target, target_mean, target_std) = generate_training_data(15000, &device)?;
    
    let (model, mut optimizer, varmap) = create_model_and_optimizer(&device, 0.01)?;
    
    let loss_history = train_model(&model, &mut optimizer, &distances, &target, target_mean, target_std, 5000, &device)?;
    
    let model_path = format!("{}/trained_model.safetensors", output_dir);
    let config_path = format!("{}/trained_model_config.json", output_dir);
    save_model_with_config(&varmap, &model_path, &config_path, target_mean, target_std)?;
    
    // Testler ve Grafikler...
    let (test_distances, cornell_values, nn_values, theory_pts, nn_pts) = test_model(&model, target_mean, target_std, &device)?;
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plot_results(&output_dir, &loss_history, &test_distances, &cornell_values, &nn_values, &model, target_mean, target_std, &device)
    }));

    // Statik Simülasyon (Rapor için)
    let sc_params = ScatteringParams::default();
    let electrons = simulate_scattering(&model, &sc_params, target_mean, target_std, &device)?;
    let scattering_file = format!("{}/scattering.svg", output_dir);
    plot_scattering(&electrons, &scattering_file);

    // GUI Verileri
    let electron_data: Vec<gui::ElectronData> = electrons.iter().map(|e| gui::ElectronData {
        trajectory: e.trajectory.clone(),
        impact_parameter: e.impact_parameter,
    }).collect();

    let app_data = AppData {
        loss_history,
        potential_theory: theory_pts,
        potential_nn: nn_pts,
        test_distances,
        cornell_values,
        nn_values,
        loss_file: "Generated".to_string(),
        potential_file: "Generated".to_string(),
        scattering_file: Some(scattering_file),
        electrons: Some(electron_data),
    };

    app_data.save_session(&output_dir).unwrap();
    
    // CANLI MOD İÇİN CONTEXT HAZIRLA
    // Modeli Arc ile sarmalıyoruz ki GUI thread'i ile paylaşabilelim
    let interactive_ctx = InteractiveContext {
        model: Arc::new(model), // Modeli paylaşıma aç
        device: device,
        mean: target_mean,
        std: target_std,
        live_electrons: Vec::new(),
        targets: get_proton_quarks(),
    };

    println!("✅ Simülasyon hazır! GUI açılıyor...");
    launch_gui(app_data, "Cornell Laboratuvarı", Some(interactive_ctx));

    Ok(())
}

fn run_with_pretrained_model(model_path: &str, config_path: &str) -> Result<()> {
    println!("🚀 EĞİTİLMİŞ MODEL MODU");
    let device = Device::Cpu;
    
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S_LOADED").to_string();
    let output_dir = format!("outputs/{}", timestamp);
    std::fs::create_dir_all(&output_dir).expect("Klasör yok");

    let (model, _varmap, target_mean, target_std) = load_model_with_config(model_path, config_path, &device)?;
    
    // Test ve Statik Simülasyon
    let (test_distances, cornell_values, nn_values, theory_pts, nn_pts) = test_model(&model, target_mean, target_std, &device)?;
    let sc_params = ScatteringParams::default();
    let electrons = simulate_scattering(&model, &sc_params, target_mean, target_std, &device)?;
    let scattering_file = format!("{}/scattering.svg", output_dir);
    plot_scattering(&electrons, &scattering_file);
    
    let electron_data: Vec<gui::ElectronData> = electrons.iter().map(|e| gui::ElectronData {
        trajectory: e.trajectory.clone(),
        impact_parameter: e.impact_parameter,
    }).collect();

    let app_data = AppData {
        loss_history: vec![],
        potential_theory: theory_pts,
        potential_nn: nn_pts,
        test_distances,
        cornell_values,
        nn_values,
        loss_file: "Loaded".to_string(),
        potential_file: "Loaded".to_string(),
        scattering_file: Some(scattering_file),
        electrons: Some(electron_data),
    };
    
    // CANLI MOD CONTEXT
    let interactive_ctx = InteractiveContext {
        model: Arc::new(model),
        device,
        mean: target_mean,
        std: target_std,
        live_electrons: Vec::new(),
        targets: get_proton_quarks(),
    };
    
    launch_gui(app_data, "Cornell Laboratuvarı (Eğitilmiş)", Some(interactive_ctx));
    Ok(())
}