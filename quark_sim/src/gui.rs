// GUI ve İnteraktif Simülasyon

use eframe::egui;
use egui_plot::{Line, PlotPoints, Plot, Points};
use serde::{Serialize, Deserialize};
use std::fs;
use std::sync::Arc; // Modeli threadler arası paylaşmak için
use candle_core::Device;
use crate::model::QuarkModel;
use crate::scattering::{Electron, get_proton_quarks, TargetQuark};

/// Elektron yörüngesi (Veri saklama için)
#[derive(Serialize, Deserialize, Clone)]
pub struct ElectronData {
    pub trajectory: Vec<(f32, f32)>,
    pub impact_parameter: f32,
}

/// GUI veri yapısı (Save/Load için)
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

/// Canlı Simülasyon için Gerekli Veriler (Model vb.)
/// Bu kısım JSON'a kaydedilmez, çalışma anında oluşur.
pub struct InteractiveContext {
    pub model: Arc<QuarkModel>,
    pub device: Device,
    pub mean: f32,
    pub std: f32,
    pub live_electrons: Vec<Electron>, // Canlı uçan elektronlar
    pub targets: Vec<TargetQuark>,
}

pub struct SimulationApp {
    data: AppData,
    interactive: Option<InteractiveContext>, // Varsa interaktif mod çalışır
    show_loss: bool,
    show_theory: bool,
    show_nn: bool,
    show_points: bool,
}

impl SimulationApp {
    pub fn new(data: AppData, interactive: Option<InteractiveContext>) -> Self {
        Self {
            data,
            interactive,
            show_loss: true,
            show_theory: true,
            show_nn: true,
            show_points: true,
        }
    }
}

impl eframe::App for SimulationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Canlı Fizik Döngüsü (Eğer interaktif moddaysak)
        if let Some(ref mut context) = self.interactive {
            // Her karede elektronları biraz ilerlet
            let dt = 0.05;
            let force_scale = 0.2;
            
            for e in &mut context.live_electrons {
                // Ekrandan çok çıkmadığı sürece güncelle
                if e.x < 10.0 && e.x > -10.0 && e.y.abs() < 8.0 {
                    let _ = e.update_step(
                        &context.model, 
                        &context.targets, 
                        context.mean, 
                        context.std, 
                        &context.device, 
                        dt, 
                        force_scale
                    );
                }
            }
            // Sürekli güncelleme iste (Animasyon için)
            ctx.request_repaint(); 
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("🔬 Cornell Potansiyeli ve İnteraktif Proton Laboratuvarı");
                
                if self.interactive.is_some() {
                    ui.colored_label(egui::Color32::GREEN, "🟢 CANLI MOD AKTİF: Grafiğe tıklayarak elektron ateşleyin!");
                } else {
                    ui.label("🔴 Sadece İzleme Modu (Model yüklü değil)");
                }

                ui.separator();
                
                // --- İNTERAKTİF GRAFİK ALANI ---
                ui.group(|ui| {
                    ui.heading("⚛️ Proton Çarpıştırıcısı (Tıkla ve Ateşle!)");
                    
                    let plot = Plot::new("interactive_plot")
                        .height(500.0)
                        .data_aspect(1.0)
                        .allow_drag(true)
                        .allow_zoom(true)
                        .coordinates_formatter(egui_plot::Corner::LeftBottom, egui_plot::CoordinatesFormatter::default());

                    plot.show(ui, |plot_ui| {
                        // 1. Hedef Kuarkları Çiz
                        let targets = if let Some(ref ctx) = self.interactive {
                            ctx.targets.clone()
                        } else {
                            get_proton_quarks()
                        };

                        let quark_points: PlotPoints = targets.iter()
                            .map(|q| [q.x as f64, q.y as f64])
                            .collect();
                        
                        plot_ui.points(Points::new(quark_points).radius(8.0).color(egui::Color32::RED).name("Kuarklar"));

                        // 2. Geçmiş Elektron Yörüngelerini Çiz (Statik)
                        if let Some(ref electrons) = self.data.electrons {
                             for (i, e) in electrons.iter().enumerate() {
                                let points: PlotPoints = e.trajectory.iter().map(|(x, y)| [*x as f64, *y as f64]).collect();
                                plot_ui.line(Line::new(points).color(egui::Color32::from_gray(100)).width(1.0).name(format!("Geçmiş {}", i)));
                             }
                        }

                        // 3. CANLI Elektronları Çiz
                        if let Some(ref ctx) = self.interactive {
                            for (i, e) in ctx.live_electrons.iter().enumerate() {
                                let points: PlotPoints = e.trajectory.iter().map(|(x, y)| [*x as f64, *y as f64]).collect();
                                // Canlı elektronlar parlak sarı olsun
                                plot_ui.line(Line::new(points).color(egui::Color32::YELLOW).width(2.5).name(format!("Canlı {}", i)));
                                // Başını nokta olarak koy
                                plot_ui.points(Points::new(vec![[e.x as f64, e.y as f64]]).radius(4.0).color(egui::Color32::YELLOW));
                            }
                        }

                        // 4. MOUSE TIKLAMASI İLE ATEŞLEME
                        if self.interactive.is_some() && plot_ui.response().clicked() {
                            // Tıklanan koordinatları al
                            if let Some(pointer_pos) = plot_ui.pointer_coordinate() {
                                let x = pointer_pos.x as f32;
                                let y = pointer_pos.y as f64 as f32; // f64 -> f32
                                
                                // Yeni elektron oluştur
                                // Tıkladığın yerden, sağa doğru (veya merkeze doğru) fırlat
                                // Angry Birds tarzı: Tıkladığın yer başlangıç, hız sabit (0.5)
                                let vx = 0.5; 
                                let vy = 0.0; // Düz fırlat, fizik onu bükecek
                                
                                let new_electron = Electron::new(x, y, vx, vy);
                                
                                if let Some(ref mut ctx) = self.interactive {
                                    ctx.live_electrons.push(new_electron);
                                }
                            }
                        }
                    });
                    
                    ui.label("İpucu: Mouse ile grafiğin herhangi bir yerine tıklayın. Elektron oradan doğacak ve sağa doğru uçarken kuarklara çarpıp saçılacak.");
                    if ui.button("Temizle (Canlı Parçacıklar)").clicked() {
                        if let Some(ref mut ctx) = self.interactive {
                            ctx.live_electrons.clear();
                        }
                    }
                });
                
                ui.separator();

                // --- İSTATİSTİK GRAFİKLERİ (ESKİ KISIM) ---
                ui.collapsing("📉 Eğitim ve Potansiyel Grafikleri", |ui| {
                     ui.columns(2, |columns| {
                        // Sol: Eğitim Kaybı
                        columns[0].group(|ui| {
                            ui.heading("Eğitim Kaybı");
                            ui.checkbox(&mut self.show_loss, "Göster");
                            if self.show_loss {
                                let points: PlotPoints = self.data.loss_history.iter().map(|(e, l)| [*e as f64, *l as f64]).collect();
                                Plot::new("loss").height(200.0).show(ui, |p| p.line(Line::new(points)));
                            }
                        });
                        // Sağ: Potansiyel
                        columns[1].group(|ui| {
                            ui.heading("Cornell Potansiyeli");
                            Plot::new("potential").height(200.0).show(ui, |p| {
                                if self.show_theory {
                                    let pts: PlotPoints = self.data.potential_theory.iter().map(|(x,y)| [*x as f64, *y as f64]).collect();
                                    p.line(Line::new(pts).color(egui::Color32::BLUE).name("Teori"));
                                }
                                if self.show_nn {
                                    let pts: PlotPoints = self.data.potential_nn.iter().map(|(x,y)| [*x as f64, *y as f64]).collect();
                                    p.line(Line::new(pts).color(egui::Color32::RED).name("Yapay Zeka"));
                                }
                            });
                        });
                     });
                });
            });
        });
    }
}

pub fn launch_gui(app_data: AppData, title: &str, interactive: Option<InteractiveContext>) {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 900.0])
            .with_title(title),
        ..Default::default()
    };
    
    eframe::run_native(
        title,
        native_options,
        Box::new(|_cc| Ok(Box::new(SimulationApp::new(app_data, interactive)))),
    ).unwrap();
}