// Fizik sabitleri ve Cornell potansiyeli

// FİZİK SABİTLERİ (Cornell Potansiyeli)
pub const ALPHA_S: f32 = 0.5;        // Güçlü etkileşim sabiti
pub const STRING_TENSION: f32 = 0.9; // String tension (GeV/fm)
pub const HBARC: f32 = 0.1973;       // ℏc (GeV·fm)

/// Cornell Potansiyeli: V(r) = -4αs/(3r) + kr
/// Kuark-antikuark etkileşimini modeller
pub fn cornell_potential(r: f32) -> f32 {
    let coulomb = -(4.0 * ALPHA_S) / (3.0 * r); // Kısa mesafe çekimi
    let linear = STRING_TENSION * r;             // Lineer hapsetme
    coulomb + linear
}
