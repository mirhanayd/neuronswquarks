// Deep Inelastic Scattering (DIS) Simülasyonu
// Elektronların kuark hedefine saçılması

use candle_core::{Device, Result, Tensor};
use crate::model::QuarkModel;

/// Elektron yapısı
#[derive(Clone, Debug)]
pub struct Electron {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub trajectory: Vec<(f32, f32)>,
    pub impact_parameter: f32, // Başlangıç y konumu (merkeze uzaklık)
}

impl Electron {
    /// Yeni elektron oluştur (soldan fırlatılır)
    pub fn new(impact_parameter: f32, initial_velocity: f32) -> Self {
        Self {
            x: -5.0, // Sol taraftan başla
            y: impact_parameter,
            vx: initial_velocity,
            vy: 0.0,
            trajectory: vec![(-5.0, impact_parameter)],
            impact_parameter,
        }
    }
    
    /// Elektronun merkezden uzaklığını hesapla
    pub fn distance_from_center(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// DIS simülasyonu parametreleri
pub struct ScatteringParams {
    pub num_electrons: usize,
    pub max_impact_parameter: f32,
    pub initial_velocity: f32,
    pub time_step: f32,
    pub max_steps: usize,
    pub force_scale: f32, // Kuvvet ölçeklendirme faktörü
}

impl Default for ScatteringParams {
    fn default() -> Self {
        Self {
            num_electrons: 20,
            max_impact_parameter: 2.0, // -2.0 ile +2.0 fm arası
            initial_velocity: 0.5, // c cinsinden
            time_step: 0.05,
            max_steps: 300,
            force_scale: 0.1,
        }
    }
}

/// Deep Inelastic Scattering simülasyonu
pub fn simulate_scattering(
    model: &QuarkModel,
    params: &ScatteringParams,
    target_mean: f32,
    target_std: f32,
    device: &Device,
) -> Result<Vec<Electron>> {
    let mut electrons = Vec::new();
    
    println!("\n⚛️ Deep Inelastic Scattering Simülasyonu / DIS Simulation");
    println!("   {} elektron fırlatılıyor / Firing {} electrons", params.num_electrons, params.num_electrons);
    println!("   İmpact parametresi / Impact parameter: ±{:.2} fm", params.max_impact_parameter);
    
    // Elektronları farklı impact parametreleri ile oluştur
    for i in 0..params.num_electrons {
        let impact = -params.max_impact_parameter + 
            (2.0 * params.max_impact_parameter * i as f32) / (params.num_electrons - 1) as f32;
        
        let mut electron = Electron::new(impact, params.initial_velocity);
        
        // Elektronun yörüngesini simüle et
        for _step in 0..params.max_steps {
            let r = electron.distance_from_center();
            
            // Elektron çok uzaklaştıysa simülasyonu bitir
            if electron.x > 5.0 || r > 6.0 {
                break;
            }
            
            // Yapay zekaya sor: Bu noktada potansiyel enerji ne kadar?
            let input = Tensor::from_vec(
                vec![electron.x, electron.y, 0.0],
                (1, 3),
                device,
            )?;
            
            let potential_norm = model.forward(&input)?;
            let potential = potential_norm
                .broadcast_mul(&Tensor::new(&[target_std], device)?)?
                .broadcast_add(&Tensor::new(&[target_mean], device)?)?;
            let V = potential.to_vec2::<f32>()?[0][0];
            
            // Kuvvet = -∇V (gradient of potential)
            // Basit sayısal türev: F = -dV/dr
            let epsilon = 0.01;
            
            // x yönünde gradient
            let x_plus = electron.x + epsilon;
            let input_x = Tensor::from_vec(vec![x_plus, electron.y, 0.0], (1, 3), device)?;
            let V_x = model.forward(&input_x)?
                .broadcast_mul(&Tensor::new(&[target_std], device)?)?
                .broadcast_add(&Tensor::new(&[target_mean], device)?)?
                .to_vec2::<f32>()?[0][0];
            let fx = -(V_x - V) / epsilon * params.force_scale;
            
            // y yönünde gradient
            let y_plus = electron.y + epsilon;
            let input_y = Tensor::from_vec(vec![electron.x, y_plus, 0.0], (1, 3), device)?;
            let V_y = model.forward(&input_y)?
                .broadcast_mul(&Tensor::new(&[target_std], device)?)?
                .broadcast_add(&Tensor::new(&[target_mean], device)?)?
                .to_vec2::<f32>()?[0][0];
            let fy = -(V_y - V) / epsilon * params.force_scale;
            
            // Hız güncelleme (F = ma, m = 1 varsayımı)
            electron.vx += fx * params.time_step;
            electron.vy += fy * params.time_step;
            
            // Konum güncelleme
            electron.x += electron.vx * params.time_step;
            electron.y += electron.vy * params.time_step;
            
            // Yörüngeye ekle
            electron.trajectory.push((electron.x, electron.y));
        }
        
        electrons.push(electron);
    }
    
    println!("   ✓ Saçılma simülasyonu tamamlandı / Scattering simulation completed");
    
    // İstatistikler
    let mut large_angle = 0;
    let mut small_angle = 0;
    for e in &electrons {
        let final_angle = (e.vy / e.vx).atan().abs().to_degrees();
        if final_angle > 10.0 {
            large_angle += 1;
        } else {
            small_angle += 1;
        }
    }
    
    println!("\n   📊 Saçılma İstatistikleri / Scattering Statistics:");
    println!("      • Geniş açı / Large angle (>10°): {} elektron", large_angle);
    println!("      • Küçük açı / Small angle (<10°): {} elektron", small_angle);
    
    Ok(electrons)
}

/// Saçılma grafiğini çiz
pub fn plot_scattering(
    electrons: &[Electron],
    filename: &str,
) {
    use plotters::prelude::*;
    
    let root = SVGBackend::new(filename, (1000, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Deep Inelastic Scattering - Elektron Yörüngeleri / Electron Trajectories", 
                 ("sans-serif", 40))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(-6f32..6f32, -3f32..3f32)
        .unwrap();
    
    chart.configure_mesh()
        .x_desc("x (fm)")
        .y_desc("y (fm)")
        .draw()
        .unwrap();
    
    // Merkezdeki kuark hedefini çiz (kırmızı daire)
    chart.draw_series(std::iter::once(Circle::new(
        (0.0, 0.0),
        10,
        RED.filled(),
    ))).unwrap()
        .label("Kuark Hedefi / Quark Target")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.filled()));
    
    // Her elektronun yörüngesini çiz
    let colors = [
        &BLUE, &GREEN, &RED, &CYAN, &MAGENTA, &YELLOW,
        &BLACK, &RGBColor(128, 0, 128), &RGBColor(255, 165, 0),
    ];
    
    for (i, electron) in electrons.iter().enumerate() {
        let color = colors[i % colors.len()];
        
        chart.draw_series(LineSeries::new(
            electron.trajectory.iter().map(|&(x, y)| (x, y)),
            color.stroke_width(2),
        )).unwrap();
        
        // Başlangıç noktası
        chart.draw_series(std::iter::once(Circle::new(
            electron.trajectory[0],
            4,
            color.filled(),
        ))).unwrap();
    }
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    
    root.present().unwrap();
    println!("   ✓ {} kaydedildi / saved", filename);
}
