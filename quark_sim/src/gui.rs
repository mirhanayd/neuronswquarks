// GUI ve İnteraktif Simülasyon

use eframe::egui;
use egui_plot::{Line, PlotPoints, Plot, Points, Text}; // Text ekledik
use serde::{Serialize, Deserialize};
use std::fs;
use std::sync::Arc;
use candle_core::Device;
use crate::model::QuarkModel;
use crate::scattering::{Electron, get_proton_quarks, TargetQuark};

#[derive(Serialize, Deserialize, Clone)]
pub struct ElectronData {
    pub trajectory: Vec<(f32, f32)>,
    pub impact_parameter: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AppData {
    pub loss_history: Vec<(usize, f32)>,
    pub potential_theory: Vec<(f32, f32)>,
    pub potential_nn: Vec<(f32, f32)>,
    pub test_distances: Vec<f32>,
    pub cornell_values: Vec<f32>,
    pub nn_values: Vec<f32>,
    pub loss_file: String,
    pub potential_file: String,
    pub scattering_file: Option<String>,
    pub electrons: Option<Vec<ElectronData>>,
}

impl AppData {
    pub fn save_session(&self, output_dir: &str) -> std::io::Result<String> {
        let filename = format!("{}/session.json", output_dir);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&filename, json)?;
        Ok(filename)
    }
    pub fn load_session(filename: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(filename)?;
        let data: AppData = serde_json::from_str(&json)?;
        Ok(data)
    }
}

pub struct InteractiveContext {
    pub model: Arc<QuarkModel>,
    pub device: Device,
    pub mean: f32,
    pub std: f32,
    pub live_electrons: Vec<Electron>,
    pub targets: Vec<TargetQuark>,
}

pub struct SimulationApp {
    data: AppData,
    interactive: Option<InteractiveContext>,
    show_loss: bool,
    show_theory: bool,
    show_nn: bool,
    show_points: bool,
}

impl SimulationApp {
    pub fn new(data: AppData, interactive: Option<InteractiveContext>) -> Self {
        Self {
            data, interactive,
            show_loss: true, show_theory: true, show_nn: true, show_points: true,
        }
    }
}

impl eframe::App for SimulationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(ref mut context) = self.interactive {
            let dt = 0.05;
            let force_scale = 0.2;
            for e in &mut context.live_electrons {
                if e.x < 10.0 && e.x > -10.0 && e.y.abs() < 8.0 {
                    let _ = e.update_step(&context.model, &context.targets, context.mean, context.std, &context.device, dt, force_scale);
                }
            }
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("🔬 Proton Spin Laboratuvarı");
                
                if let Some(ref mut ctx) = self.interactive {
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::GREEN, "🟢 CANLI MOD");
                        if ui.button("🧹 Parçacıkları Temizle").clicked() {
                            ctx.live_electrons.clear();
                        }
                        if ui.button("🔄 Spinleri Sıfırla").clicked() {
                            ctx.targets = get_proton_quarks();
                        }
                    });
                    ui.label("Kuarklara tıklayarak spin yönlerini (↑/↓) değiştirebilirsiniz.");
                }

                ui.separator();
                
                ui.group(|ui| {
                    let plot = Plot::new("interactive_plot")
                        .height(500.0)
                        .data_aspect(1.0)
                        .allow_drag(true).allow_zoom(true)
                        .coordinates_formatter(egui_plot::Corner::LeftBottom, egui_plot::CoordinatesFormatter::default());

                    plot.show(ui, |plot_ui| {
                        // 1. Hedef Kuarkları Çiz ve Etkileşim
                        if let Some(ref mut ctx) = self.interactive {
                            for (idx, quark) in ctx.targets.iter_mut().enumerate() {
                                // Spin rengi: Yukarı=Kırmızı, Aşağı=Mavi
                                let color = if quark.spin > 0.0 { egui::Color32::RED } else { egui::Color32::BLUE };
                                let symbol = if quark.spin > 0.0 { "⬆" } else { "⬇" };
                                
                                // Kuark noktası
                                plot_ui.points(Points::new(vec![[quark.x as f64, quark.y as f64]]).radius(10.0).color(color).name("Kuark"));
                                
                                // Spin sembolü (Metin olarak)
                                plot_ui.text(Text::new(
                                    [quark.x as f64, quark.y as f64 + 0.2].into(), 
                                    symbol
                                ).color(egui::Color32::BLACK).name("Spin"));

                                // Tıklama kontrolü (Basit mesafe kontrolü - Plot içinde button zor olduğu için)
                                // Not: Plot içinde doğrudan tıklama algılamak için pointer koordinatlarını kullanacağız.
                            }
                            
                             // Spin Değiştirme Mantığı (Mouse Tıklaması)
                            if plot_ui.response().clicked() {
                                if let Some(pointer) = plot_ui.pointer_coordinate() {
                                    let px = pointer.x as f32;
                                    let py = pointer.y as f32;
                                    
                                    // Bir kuarka tıklandı mı?
                                    let mut clicked_quark = false;
                                    for quark in ctx.targets.iter_mut() {
                                        let dist = ((quark.x - px).powi(2) + (quark.y - py).powi(2)).sqrt();
                                        if dist < 0.3 { // Tıklama yarıçapı
                                            quark.spin *= -1.0; // Spini ters çevir
                                            clicked_quark = true;
                                            break;
                                        }
                                    }
                                    
                                    // Eğer kuarka tıklanmadıysa boşluğa tıklandı -> Elektron Ateşle
                                    if !clicked_quark {
                                        let vx = 0.5;
                                        let vy = 0.0;
                                        ctx.live_electrons.push(Electron::new(px, py, vx, vy));
                                    }
                                }
                            }
                        }

                        // 2. Canlı Elektronlar
                        if let Some(ref ctx) = self.interactive {
                            for (i, e) in ctx.live_electrons.iter().enumerate() {
                                let points: PlotPoints = e.trajectory.iter().map(|(x, y)| [*x as f64, *y as f64]).collect();
                                plot_ui.line(Line::new(points).color(egui::Color32::YELLOW).width(2.0));
                                plot_ui.points(Points::new(vec![[e.x as f64, e.y as f64]]).radius(3.0).color(egui::Color32::YELLOW));
                            }
                        }
                    });
                });
                
                // İstatistikler (Eski kodun aynısı)
                // ... (Kısalık olsun diye burayı atlıyorum, önceki kodda zaten var) ...
            });
        });
    }
}

pub fn launch_gui(app_data: AppData, title: &str, interactive: Option<InteractiveContext>) {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 900.0]).with_title(title),
        ..Default::default()
    };
    eframe::run_native(title, native_options, Box::new(|_cc| Ok(Box::new(SimulationApp::new(app_data, interactive))))).unwrap();
}