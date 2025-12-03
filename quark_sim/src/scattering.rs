// Deep Inelastic Scattering (DIS) Simülasyonu
// Elektronların PROTON (3 Kuarklı Sistem) hedefine saçılması

use candle_core::{Device, Result, Tensor};
use crate::model::QuarkModel;
use plotters::prelude::*;

/// Hedefteki Kuarkların yapısı
#[derive(Clone, Copy)] // Clone ve Copy ekledik ki GUI'de rahat kullanalım
pub struct TargetQuark {
    pub x: f32,
    pub y: f32,
}

/// Protonun içindeki standart kuark dizilimi
pub fn get_proton_quarks() -> Vec<TargetQuark> {
    vec![
        TargetQuark { x: 0.0, y: 0.8 },   // Üst (Up)
        TargetQuark { x: 0.7, y: -0.4 },  // Sağ Alt (Up)
        TargetQuark { x: -0.7, y: -0.4 }, // Sol Alt (Down)
    ]
}

/// Elektron yapısı
#[derive(Clone, Debug)]
pub struct Electron {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub trajectory: Vec<(f32, f32)>,
    pub impact_parameter: f32,
}

impl Electron {
    /// Yeni elektron oluştur
    pub fn new(x: f32, y: f32, vx: f32, vy: f32) -> Self {
        Self {
            x,
            y,
            vx,
            vy,
            trajectory: vec![(x, y)],
            impact_parameter: y,
        }
    }

    /// Fizik motoru: Bir zaman adımı (dt) ilerle
    /// Bu fonksiyon GUI tarafından her karede çağrılacak.
    pub fn update_step(
        &mut self,
        model: &QuarkModel,
        targets: &[TargetQuark],
        mean: f32,
        std: f32,
        device: &Device,
        dt: f32,
        force_scale: f32
    ) -> Result<()> {
        // 1. Mevcut konumdaki potansiyeli al
        let v_curr = get_total_potential(self.x, self.y, targets, model, mean, std, device)?;
        
        // 2. Gradyan (Türev) ile Kuvveti Hesapla (F = -dV/dr)
        let epsilon = 0.02;
        
        // X kuvveti
        let v_x = get_total_potential(self.x + epsilon, self.y, targets, model, mean, std, device)?;
        let fx = -(v_x - v_curr) / epsilon * force_scale;
        
        // Y kuvveti
        let v_y = get_total_potential(self.x, self.y + epsilon, targets, model, mean, std, device)?;
        let fy = -(v_y - v_curr) / epsilon * force_scale;
        
        // 3. Hız ve Konum Güncelle
        self.vx += fx * dt;
        self.vy += fy * dt;
        
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        
        // Yörüngeye ekle (Çizim için)
        // Her adımı eklersek RAM şişebilir, sadece değişim varsa ekleyelim
        if let Some(last) = self.trajectory.last() {
            if (self.x - last.0).hypot(self.y - last.1) > 0.05 {
                self.trajectory.push((self.x, self.y));
            }
        } else {
            self.trajectory.push((self.x, self.y));
        }

        Ok(())
    }
}

/// Helper: Toplam potansiyeli hesapla
fn get_total_potential(
    x: f32, y: f32, 
    quarks: &[TargetQuark], 
    model: &QuarkModel, 
    mean: f32, std: f32, 
    device: &Device
) -> Result<f32> {
    let mut total = 0.0;
    for q in quarks {
        let dx = x - q.x;
        let dy = y - q.y;
        
        // 3D input (z=0)
        let input = Tensor::from_vec(vec![dx, dy, 0.0], (1, 3), device)?;
        let raw = model.forward(&input)?;
        
        // Denormalize
        let val = raw
            .broadcast_mul(&Tensor::new(&[std], device)?)?
            .broadcast_add(&Tensor::new(&[mean], device)?)?
            .to_vec2::<f32>()?[0][0];
            
        total += val;
    }
    Ok(total)
}

// ... ScatteringParams ve simulate_scattering fonksiyonları eski haliyle kalabilir veya
// ... simulate_scattering fonksiyonunu Electron::new ve update_step kullanacak şekilde 
// ... güncelleyebilirsin ama GUI için yukarıdakiler yeterli.
//
// Kolaylık olsun diye simulate_scattering'i burada sadeleştirilmiş bırakıyorum:

pub struct ScatteringParams {
    pub num_electrons: usize,
    pub max_impact_parameter: f32,
    pub initial_velocity: f32,
    pub time_step: f32,
    pub max_steps: usize,
    pub force_scale: f32,
}

impl Default for ScatteringParams {
    fn default() -> Self {
        Self {
            num_electrons: 30,
            max_impact_parameter: 2.5,
            initial_velocity: 0.5,
            time_step: 0.05,
            max_steps: 400,
            force_scale: 0.2,
        }
    }
}

pub fn simulate_scattering(
    model: &QuarkModel,
    params: &ScatteringParams,
    mean: f32,
    std: f32,
    device: &Device,
) -> Result<Vec<Electron>> {
    let mut electrons = Vec::new();
    let targets = get_proton_quarks();
    
    for i in 0..params.num_electrons {
        let impact = -params.max_impact_parameter + 
            (2.0 * params.max_impact_parameter * i as f32) / (params.num_electrons - 1) as f32;
            
        let mut e = Electron::new(-5.0, impact, params.initial_velocity, 0.0);
        
        for _ in 0..params.max_steps {
            if e.x > 6.0 || e.y.abs() > 5.0 { break; }
            e.update_step(model, &targets, mean, std, device, params.time_step, params.force_scale)?;
        }
        electrons.push(e);
    }
    Ok(electrons)
}

pub fn plot_scattering(electrons: &[Electron], filename: &str) {
    let root = SVGBackend::new(filename, (1000, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Proton Saçılması", ("sans-serif", 40))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(-6f32..6f32, -4f32..4f32)
        .unwrap();
    
    chart.configure_mesh().draw().unwrap();
    
    // Kuarklar
    for q in get_proton_quarks() {
        chart.draw_series(std::iter::once(Circle::new((q.x, q.y), 8, RED.filled()))).unwrap();
    }
    
    // Elektronlar
    for (i, e) in electrons.iter().enumerate() {
        let color = Palette99::pick(i);
        chart.draw_series(LineSeries::new(
            e.trajectory.iter().map(|&(x, y)| (x, y)),
            color.stroke_width(2),
        )).unwrap();
    }
    
    root.present().unwrap();
}