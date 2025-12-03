// GUI bileşenleri

use eframe::egui;
use egui_plot::{Line, PlotPoints};
use serde::{Serialize, Deserialize};
use std::fs;

/// Elektron yörüngesi (DIS için)
#[derive(Serialize, Deserialize, Clone)]
pub struct ElectronData {
    pub trajectory: Vec<(f32, f32)>,
    pub impact_parameter: f32,
}

/// GUI için veri yapısı
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
    /// Oturum verilerini JSON dosyasına kaydet
    pub fn save_session(&self, output_dir: &str) -> std::io::Result<String> {
        let filename = format!("{}/session.json", output_dir);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&filename, json)?;
        Ok(filename)
    }
    
    /// JSON dosyasından oturum verilerini yükle
    pub fn load_session(filename: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(filename)?;
        let data: AppData = serde_json::from_str(&json)?;
        Ok(data)
    }
}

/// GUI uygulaması
pub struct SimulationApp {
    data: AppData,
    show_loss: bool,
    show_theory: bool,
    show_nn: bool,
    show_points: bool,
}

impl SimulationApp {
    pub fn new(data: AppData) -> Self {
        Self {
            data,
            show_loss: true,
            show_theory: true,
            show_nn: true,
            show_points: true,
        }
    }
}

impl eframe::App for SimulationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🔬 Cornell Potansiyeli - Kuark Simülasyonu");
            ui.separator();
            
            // Dosya bilgileri
            ui.horizontal(|ui| {
                ui.label("📁 Kayıtlı dosyalar:");
                ui.label(&self.data.loss_file);
                ui.label(&self.data.potential_file);
                if let Some(ref scattering) = self.data.scattering_file {
                    ui.label(scattering);
                }
            });
            
            ui.separator();
            
            // İki sütunlu layout (Eğitim + Potansiyel)
            ui.columns(2, |columns| {
                // Sol sütun: Eğitim Kaybı
                columns[0].group(|ui| {
                    ui.heading("Eğitim Kaybı");
                    ui.checkbox(&mut self.show_loss, "Göster");
                    
                    if self.show_loss {
                        let loss_points: PlotPoints = self.data.loss_history.iter()
                            .map(|(e, l)| [*e as f64, *l as f64])
                            .collect();
                        
                        egui_plot::Plot::new("loss_plot")
                            .height(300.0)
                            .show(ui, |plot_ui| {
                                plot_ui.line(Line::new(loss_points).name("Kayıp (MSE)"));
                            });
                    }
                });
                
                // Sağ sütun: Cornell Potansiyeli
                columns[1].group(|ui| {
                    ui.heading("Cornell Potansiyeli");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.show_theory, "Teori");
                        ui.checkbox(&mut self.show_nn, "NN Tahmini");
                        ui.checkbox(&mut self.show_points, "Test Noktaları");
                    });
                    
                    egui_plot::Plot::new("potential_plot")
                        .height(300.0)
                        .show(ui, |plot_ui| {
                            if self.show_theory {
                                let theory_points: PlotPoints = self.data.potential_theory.iter()
                                    .map(|(r, v)| [*r as f64, *v as f64])
                                    .collect();
                                plot_ui.line(Line::new(theory_points).name("Teorik Cornell").color(egui::Color32::BLUE));
                            }
                            
                            if self.show_nn {
                                let nn_points: PlotPoints = self.data.potential_nn.iter()
                                    .map(|(r, v)| [*r as f64, *v as f64])
                                    .collect();
                                plot_ui.line(Line::new(nn_points).name("Sinir Ağı").color(egui::Color32::RED));
                            }
                            
                            if self.show_points {
                                let test_points: PlotPoints = self.data.test_distances.iter()
                                    .zip(self.data.nn_values.iter())
                                    .map(|(r, v)| [*r as f64, *v as f64])
                                    .collect();
                                plot_ui.points(egui_plot::Points::new(test_points).name("Test Noktaları").color(egui::Color32::GREEN));
                            }
                        });
                });
            });
            
            ui.separator();
            
            // Alt bölüm: Test Sonuçları Tablosu
            ui.collapsing("📊 Detaylı Test Sonuçları", |ui| {
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        egui::Grid::new("test_results_grid")
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Mesafe (fm)");
                                ui.label("Cornell (GeV)");
                                ui.label("NN (GeV)");
                                ui.label("Hata (%)");
                                ui.end_row();
                                
                                for i in 0..self.data.test_distances.len() {
                                    let r = self.data.test_distances[i];
                                    let cornell = self.data.cornell_values[i];
                                    let nn = self.data.nn_values[i];
                                    let error = ((nn - cornell).abs() / cornell.abs()) * 100.0;
                                    
                                    ui.label(format!("{:.2}", r));
                                    ui.label(format!("{:.6}", cornell));
                                    ui.label(format!("{:.6}", nn));
                                    ui.colored_label(
                                        if error < 5.0 { egui::Color32::GREEN } 
                                        else if error < 20.0 { egui::Color32::YELLOW }
                                        else { egui::Color32::RED },
                                        format!("{:.2}%", error)
                                    );
                                    ui.end_row();
                                }
                            });
                    });
            });
            
            ui.separator();
            
            // Deep Inelastic Scattering Paneli
            if let Some(ref electrons) = self.data.electrons {
                ui.group(|ui| {
                    ui.heading("⚛️ Deep Inelastic Scattering - Elektron Yörüngeleri");
                    
                    // İstatistikler
                    let mut large_angle = 0;
                    let mut small_angle = 0;
                    for e in electrons {
                        if let (Some(first), Some(last)) = (e.trajectory.first(), e.trajectory.last()) {
                            let dx = last.0 - first.0;
                            let dy = last.1 - first.1;
                            let angle = (dy / dx).atan().abs().to_degrees();
                            if angle > 10.0 {
                                large_angle += 1;
                            } else {
                                small_angle += 1;
                            }
                        }
                    }
                    
                    ui.horizontal(|ui| {
                        ui.label(format!("🎯 Toplam elektron / Total electrons: {}", electrons.len()));
                        ui.label(format!("📐 Geniş açı (>10°): {}", large_angle));
                        ui.label(format!("📏 Küçük açı (<10°): {}", small_angle));
                    });
                    
                    egui_plot::Plot::new("scattering_plot")
                        .height(400.0)
                        .width(ui.available_width())
                        .show(ui, |plot_ui| {
                            // Hedef noktası (merkez)
                            let target: PlotPoints = vec![[0.0, 0.0]].into();
                            plot_ui.points(
                                egui_plot::Points::new(target)
                                    .name("Kuark Hedefi / Quark Target")
                                    .color(egui::Color32::RED)
                                    .radius(8.0)
                            );
                            
                            // Elektron yörüngelerini çiz
                            let colors = [
                                egui::Color32::BLUE,
                                egui::Color32::GREEN,
                                egui::Color32::from_rgb(255, 165, 0), // Orange
                                egui::Color32::from_rgb(128, 0, 128), // Purple
                                egui::Color32::from_rgb(0, 128, 128), // Teal
                                egui::Color32::YELLOW,
                                egui::Color32::from_rgb(255, 20, 147), // Pink
                                egui::Color32::from_rgb(0, 255, 127), // Spring green
                            ];
                            
                            for (i, electron) in electrons.iter().enumerate() {
                                let trajectory: PlotPoints = electron.trajectory.iter()
                                    .map(|&(x, y)| [x as f64, y as f64])
                                    .collect();
                                
                                let color = colors[i % colors.len()];
                                plot_ui.line(
                                    Line::new(trajectory)
                                        .name(format!("e⁻ {}", i + 1))
                                        .color(color)
                                        .width(1.5)
                                );
                            }
                        });
                });
            }
        });
    }
}

/// GUI'yi başlat
pub fn launch_gui(app_data: AppData, title: &str) {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title(title),
        ..Default::default()
    };
    
    eframe::run_native(
        title,
        native_options,
        Box::new(|_cc| Ok(Box::new(SimulationApp::new(app_data)))),
    ).unwrap();
}
