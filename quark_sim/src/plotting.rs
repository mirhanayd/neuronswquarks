// Grafik çizim ve görselleştirme

use candle_core::{Device, Tensor};
use plotters::prelude::*;
use plotters::backend::SVGBackend;
use textplots::{Chart, Plot, Shape};

use crate::model::QuarkModel;
use crate::physics::cornell_potential;

/// SVG grafikleri oluştur ve kaydet
pub fn plot_results(
    output_dir: &str,
    loss_history: &[(usize, f32)],
    test_distances: &[f32],
    cornell_values: &[f32],
    nn_values: &[f32],
    model: &QuarkModel,
    target_mean: f32,
    target_std: f32,
    device: &Device,
) -> (String, String) {
    // Grafik 1: Eğitim Kaybı
    let loss_filename = format!("{}/training_loss.svg", output_dir);
    plot_loss_history(&loss_filename, loss_history);
    
    // Grafik 2: Cornell Potansiyeli Karşılaştırması
    let potential_filename = format!("{}/cornell_potential.svg", output_dir);
    plot_potential_comparison(
        &potential_filename,
        test_distances,
        cornell_values,
        nn_values,
        model,
        target_mean,
        target_std,
        device,
    );
    
    (loss_filename, potential_filename)
}

/// Eğitim kaybı grafiği
fn plot_loss_history(filename: &str, loss_history: &[(usize, f32)]) {
    let filename_clone = filename.to_string();
    let loss_plot = SVGBackend::new(&filename_clone, (800, 600)).into_drawing_area();
    loss_plot.fill(&WHITE).unwrap();
    
    let max_loss = loss_history.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let min_loss = loss_history.iter().map(|(_, l)| *l).fold(f32::INFINITY, f32::min);
    let max_epoch = loss_history.iter().map(|(e, _)| *e).max().unwrap_or(10000) as f32;
    
    let mut loss_chart = ChartBuilder::on(&loss_plot)
        .caption("Eğitim Kaybı (Training Loss)", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..max_epoch, (min_loss * 0.9)..(max_loss * 1.1))
        .unwrap();
    
    loss_chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("MSE (GeV²)")
        .draw()
        .unwrap();
    
    loss_chart.draw_series(LineSeries::new(
        loss_history.iter().map(|(e, l)| (*e as f32, *l)),
        &BLUE,
    )).unwrap();
    
    loss_plot.present().unwrap();
    println!("   ✓ {} kaydedildi / saved", filename);
}

/// Cornell potansiyeli karşılaştırma grafiği
fn plot_potential_comparison(
    filename: &str,
    test_distances: &[f32],
    cornell_values: &[f32],
    nn_values: &[f32],
    model: &QuarkModel,
    target_mean: f32,
    target_std: f32,
    device: &Device,
) {
    let filename_clone = filename.to_string();
    let potential_plot = SVGBackend::new(&filename_clone, (800, 600)).into_drawing_area();
    potential_plot.fill(&WHITE).unwrap();
    
    // Daha fazla nokta için düzgün eğri
    let r_values: Vec<f32> = (10..=250).map(|i| i as f32 * 0.01).collect();
    let cornell_curve: Vec<(f32, f32)> = r_values.iter()
        .map(|&r| (r, cornell_potential(r)))
        .collect();
    
    let mut nn_curve = Vec::new();
    for &r in &r_values {
        let test_input = Tensor::from_vec(vec![r, 0.0, 0.0], (1, 3), device).unwrap();
        let prediction_norm = model.forward(&test_input).unwrap();
        let prediction = prediction_norm.broadcast_mul(&Tensor::new(&[target_std], device).unwrap()).unwrap()
            .broadcast_add(&Tensor::new(&[target_mean], device).unwrap()).unwrap();
        let pred_value = prediction.to_vec2::<f32>().unwrap()[0][0];
        nn_curve.push((r, pred_value));
    }
    
    let mut potential_chart = ChartBuilder::on(&potential_plot)
        .caption("Cornell Potansiyeli: Teori vs Sinir Ağı", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..2.5f32, -4f32..3f32)
        .unwrap();
    
    potential_chart.configure_mesh()
        .x_desc("Mesafe r (fm)")
        .y_desc("Potansiyel V(r) (GeV)")
        .draw()
        .unwrap();
    
    // Teorik Cornell potansiyeli (mavi)
    potential_chart.draw_series(LineSeries::new(
        cornell_curve.iter().copied(),
        &BLUE,
    )).unwrap()
        .label("Teorik Cornell")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    
    // Sinir ağı tahmini (kırmızı)
    potential_chart.draw_series(LineSeries::new(
        nn_curve.iter().copied(),
        &RED,
    )).unwrap()
        .label("Sinir Ağı Tahmini")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    
    // Test noktaları (yeşil)
    potential_chart.draw_series(
        test_distances.iter().zip(nn_values.iter()).map(|(&r, &v)| {
            Circle::new((r, v), 3, GREEN.filled())
        })
    ).unwrap()
        .label("Test Noktaları")
        .legend(|(x, y)| Circle::new((x + 10, y), 3, GREEN.filled()));
    
    potential_chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    
    // Hata hesapla ve göster
    let mut error_sum = 0.0;
    for i in 0..test_distances.len() {
        let error = ((nn_values[i] - cornell_values[i]).abs() / cornell_values[i].abs()) * 100.0;
        error_sum += error;
    }
    let avg_error = error_sum / cornell_values.len() as f32;
    
    potential_chart.draw_series(std::iter::once(Text::new(
        format!("Ortalama Hata: {:.2}%", avg_error),
        (1.8, -3.5),
        ("sans-serif", 20).into_font().color(&BLACK),
    ))).unwrap();
    
    potential_plot.present().unwrap();
    println!("   ✓ {} kaydedildi / saved", filename);
}

/// Terminal ASCII grafikleri göster
pub fn show_terminal_plots(loss_history: &[(usize, f32)], test_distances: &[f32], cornell_values: &[f32], nn_values: &[f32]) {
    println!("\n📊 TERMINAL GRAFİKLERİ / TERMINAL PLOTS:\n");
    
    // Kayıp eğrisi
    println!("=== Eğitim Kayıbı / Training Loss ===");
    let loss_points: Vec<(f32, f32)> = loss_history.iter()
        .map(|(e, l)| (*e as f32, *l))
        .collect();
    
    let max_epoch = loss_history.iter().map(|(e, _)| *e).max().unwrap_or(10000) as f32;
    Chart::new(120, 30, 0.0, max_epoch)
        .lineplot(&Shape::Lines(&loss_points))
        .display();
    
    // Cornell potansiyeli karşılaştırması
    println!("\n=== Cornell Potansiyeli / Cornell Potential (Mavi/Blue=Teori/Theory, Kırmızı/Red=NN) ===");
    let theory_points: Vec<(f32, f32)> = test_distances.iter()
        .zip(cornell_values.iter())
        .map(|(&r, &v)| (r, v))
        .collect();
    
    let nn_points: Vec<(f32, f32)> = test_distances.iter()
        .zip(nn_values.iter())
        .map(|(&r, &v)| (r, v))
        .collect();
    
    let max_distance = test_distances.iter().fold(0.0f32, |a, &b| a.max(b));
    Chart::new(120, 30, 0.0, max_distance)
        .lineplot(&Shape::Lines(&theory_points))
        .lineplot(&Shape::Lines(&nn_points))
        .display();
}
