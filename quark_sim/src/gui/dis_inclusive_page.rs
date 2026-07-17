//! Inclusive DIS calculation page.
//!
//! Displays x, Q², y, W², parton densities, F₂, F_L, xF₃,
//! differential cross section, units, and backend metadata.

use eframe::egui;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::state::{
    BackendProcess, DisConfig, GuiError, GuiErrorCategory, InclusiveResult, ProcessStatus,
};
use super::worker::{self, WorkerHandle, WorkerMessage};

/// State for the inclusive calculation page.
pub struct InclusivePageState {
    pub calc_x: String,
    pub calc_q2: String,
    pub result: Option<InclusiveResult>,
    pub process: BackendProcess,
    pub worker: Option<WorkerHandle>,
}

impl Default for InclusivePageState {
    fn default() -> Self {
        Self {
            calc_x: "0.01".to_string(),
            calc_q2: "100.0".to_string(),
            result: None,
            process: BackendProcess::default(),
            worker: None,
        }
    }
}

/// Render the inclusive calculation page.
pub fn render_inclusive_page(
    state: &mut InclusivePageState,
    config: &DisConfig,
    errors: &mut Vec<GuiError>,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
) {
    ui.heading("📊 Inclusive DIS Calculation");
    ui.separator();

    // Poll worker for updates
    if let Some(ref handle) = state.worker {
        for msg in handle.drain() {
            match msg {
                WorkerMessage::StdoutLine(line) => {
                    state.process.stdout_lines.push(line);
                }
                WorkerMessage::StderrLine(line) => {
                    state.process.stderr_lines.push(line);
                }
                WorkerMessage::Progress(text) => {
                    state.process.progress_text = text;
                }
                WorkerMessage::Completed(code) => {
                    state.process.exit_code = Some(code);
                    state.process.status = ProcessStatus::Completed;
                    // Parse the result from stdout
                    parse_inclusive_result(state);
                }
                WorkerMessage::Failed(msg) => {
                    state.process.status = ProcessStatus::Failed;
                    errors.push(GuiError::new(
                        GuiErrorCategory::ProcessFailed,
                        msg,
                    ));
                }
            }
        }
        if state.process.status == ProcessStatus::Running {
            ctx.request_repaint();
        }
    }

    // Input fields
    egui::Grid::new("inclusive_input_grid")
        .num_columns(2)
        .spacing([20.0, 8.0])
        .show(ui, |ui| {
            ui.label("Bjorken x:");
            ui.text_edit_singleline(&mut state.calc_x);
            ui.end_row();

            ui.label("Q² [GeV²]:");
            ui.text_edit_singleline(&mut state.calc_q2);
            ui.end_row();
        });

    ui.separator();

    let is_running = state.process.status == ProcessStatus::Running;

    ui.horizontal(|ui| {
        if ui
            .add_enabled(!is_running, egui::Button::new("🔬 Calculate"))
            .clicked()
        {
            start_inclusive_calculation(state, config, errors);
        }
        if ui
            .add_enabled(is_running, egui::Button::new("⏹ Cancel"))
            .clicked()
        {
            if let Some(ref handle) = state.worker {
                handle.cancel();
            }
        }
    });

    // Status
    match state.process.status {
        ProcessStatus::Running => {
            ui.spinner();
            ui.label(&state.process.progress_text);
        }
        ProcessStatus::Failed => {
            ui.colored_label(egui::Color32::RED, "⚠ Calculation failed");
            if let Some(code) = state.process.exit_code {
                ui.label(format!("Exit code: {code}"));
            }
        }
        _ => {}
    }

    // Display results
    if let Some(ref result) = state.result {
        ui.separator();
        ui.heading("Results");

        egui::Grid::new("inclusive_result_grid")
            .num_columns(2)
            .spacing([20.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("x:");
                ui.label(format!("{:.6e}", result.x));
                ui.end_row();

                ui.label("Q² [GeV²]:");
                ui.label(format!("{:.4}", result.q2_gev2));
                ui.end_row();

                ui.label("y:");
                ui.label(format!("{:.6}", result.y));
                ui.end_row();

                ui.label("W² [GeV²]:");
                ui.label(format!("{:.4}", result.w2_gev2));
                ui.end_row();

                ui.separator();
                ui.separator();
                ui.end_row();

                ui.label("F₂:");
                ui.label(format!("{:.6e}", result.f2));
                ui.end_row();

                ui.label("F_L:");
                ui.label(format!("{:.6e}", result.fl));
                ui.end_row();

                ui.label("xF₃:");
                ui.label(format!("{:.6e}", result.xf3));
                ui.end_row();

                ui.separator();
                ui.separator();
                ui.end_row();

                ui.label("d²σ/(dx dQ²) [GeV⁻⁴]:");
                ui.label(format!("{:.6e}", result.dsigma_dxdq2_gev_m4));
                ui.end_row();

                ui.label("d²σ/(dx dQ²) [pb/GeV²]:");
                ui.label(format!("{:.6e}", result.dsigma_dxdq2_pb_gev2));
                ui.end_row();

                ui.separator();
                ui.separator();
                ui.end_row();

                ui.label("Backend:");
                ui.label(&result.backend_name);
                ui.end_row();

                ui.label("Order:");
                ui.label(&result.order);
                ui.end_row();

                ui.label("PDF Set:");
                ui.label(format!("{} (member {})", result.pdf_set, result.pdf_member));
                ui.end_row();

                ui.label("Scheme:");
                ui.label(&result.scheme);
                ui.end_row();
            });

        // Parton densities
        if !result.parton_densities.is_empty() {
            ui.separator();
            ui.collapsing("🧬 Parton Densities x·f(x, Q²)", |ui| {
                egui::Grid::new("parton_density_grid")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("PDG ID");
                        ui.strong("x·f(x, Q²)");
                        ui.end_row();

                        for (pdg_id, xf) in &result.parton_densities {
                            let name = pdg_id_name(*pdg_id);
                            ui.label(format!("{name} ({pdg_id})"));
                            ui.label(format!("{xf:.6e}"));
                            ui.end_row();
                        }
                    });
            });
        }
    }

    // Logs
    if !state.process.stdout_lines.is_empty() || !state.process.stderr_lines.is_empty() {
        ui.separator();
        ui.collapsing("📋 Process Output", |ui| {
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    for line in &state.process.stdout_lines {
                        ui.label(line);
                    }
                    for line in &state.process.stderr_lines {
                        ui.colored_label(egui::Color32::YELLOW, line);
                    }
                });
        });
    }
}

fn start_inclusive_calculation(
    state: &mut InclusivePageState,
    config: &DisConfig,
    errors: &mut Vec<GuiError>,
) {
    let x: f64 = match state.calc_x.parse() {
        Ok(v) if v > 0.0 && v < 1.0 => v,
        _ => {
            errors.push(GuiError::new(
                GuiErrorCategory::InvalidKinematics,
                "Bjorken x must be in (0, 1)",
            ));
            return;
        }
    };
    let q2: f64 = match state.calc_q2.parse() {
        Ok(v) if v > 0.0 => v,
        _ => {
            errors.push(GuiError::new(
                GuiErrorCategory::InvalidKinematics,
                "Q² must be positive",
            ));
            return;
        }
    };

    state.process.reset();
    state.process.status = ProcessStatus::Running;
    state.result = None;

    let args = super::state::build_structure_function_command(
        x,
        q2,
        &config.backend,
        &config.perturbative_order,
        &config.pdf_set,
        config.pdf_member,
        config.mu_f_over_q,
        config.mu_r_over_q,
    );

    let cancel_flag = Arc::clone(&state.process.cancel_flag);
    // We run the quark_sim binary itself as a subprocess.
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "quark_sim".to_string());

    state.worker = Some(worker::spawn_subprocess(&exe, &args, cancel_flag));
}

/// Parse inclusive result from the captured stdout lines.
fn parse_inclusive_result(state: &mut InclusivePageState) {
    // The structure-functions command outputs JSON. Try to parse it.
    let combined = state.process.stdout_lines.join("\n");

    // Try to find a JSON block in the output.
    if let Some(start) = combined.find('{') {
        if let Some(end) = combined.rfind('}') {
            let json_str = &combined[start..=end];
            if let Ok(result) = serde_json::from_str::<InclusiveResult>(json_str) {
                state.result = Some(result);
                return;
            }
        }
    }

    // Fallback: no JSON found, leave result as None.
}

/// Map PDG ID to particle name.
fn pdg_id_name(pdg_id: i32) -> &'static str {
    match pdg_id {
        1 => "d",
        -1 => "d̄",
        2 => "u",
        -2 => "ū",
        3 => "s",
        -3 => "s̄",
        4 => "c",
        -4 => "c̄",
        5 => "b",
        -5 => "b̄",
        6 => "t",
        -6 => "t̄",
        21 => "g",
        _ => "unknown",
    }
}
