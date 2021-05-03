use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::{Complex, ComplexField};
use once_cell::sync::Lazy;
use plaquette::io::*;
use plaquette::plot::{fourier::*, PlotType};
use rustfft::FftPlanner;

const DIRECTORY: &str = &"../data/set_hpc_e_n24_21_04_2021/data_temp/";

const BETA: [f64; 11] = [
    1_f64, 3_f64, 6_f64, 9_f64, 12_f64, 15_f64, 18_f64, 21_f64, 24_f64, 27_f64, 30_f64,
];

const DT: f64 = 0.000_01_f64; // prod

/*
const FFT_RESOLUTION_SIZE: f64 = 0.1_f64;
static NUMBER_OF_MEASUREMENT: Lazy<usize> =
    Lazy::new(|| ((1_f64 / DT) * 2_f64 / FFT_RESOLUTION_SIZE).ceil() as usize);

const LATTICE_DIM: usize = 24;
const LATTICE_SIZE: f64 = 1_f64;
*/

fn main() {
    let pb = ProgressBar::new(BETA.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .progress_chars("=>-")
            .template(&"{prefix:14} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>6}/{len:6} [ETA {eta_precise}] {msg}"),
    );
    pb.enable_steady_tick(499);

    for beta in &BETA {
        let file_name = format!("{}mean_measures_corr_e_{}.csv", DIRECTORY, beta);
        let result: Vec<[f64; 2]> = read_file_csv(&file_name, 1_000).unwrap();
        //let values = result.iter().map(|[x, _err]| x).copied().collect::<Vec<f64>>();
        //drop(result);

        // FFT
        let mut measure_fft = result
            .iter()
            .map(|el| el[0].into())
            .collect::<Vec<Complex<f64>>>();

        drop(result);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(measure_fft.len());

        fft.process(&mut measure_fft);
        // the /2 is that correct ?
        let normalization_fact = ((measure_fft.len() / 2) as f64).sqrt();
        let fft_real = measure_fft
            .iter()
            .take(measure_fft.len() / 2)
            .map(|val| (2_f64 * val.real()) / normalization_fact)
            .collect::<Vec<f64>>();
        drop(measure_fft);

        plot_data_fft_norm(
            &fft_real,
            DT,
            &format!("data/plot_fft_{}.svg", beta),
            PlotType::Circle,
        )
        .unwrap();
        pb.inc(1);
    }
    pb.finish();
}
