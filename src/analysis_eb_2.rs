use std::error::Error;

use lattice_qcd_rs::lattice::LatticeCyclique;
use lattice_qcd_rs::{Complex, ComplexField};
use once_cell::sync::Lazy;
use plaquette::io::read_file_csv;
//use rayon::prelude::*;
use plotters::prelude::*;
use rustfft::FftPlanner;

const DIRECTORY: &str = "../data/set_sim_eb_2_02_06_21";
const PLOT_SUB_DIR: &str = "plot";
/*
const BETA: [f64; 11] = [
    1_f64, 3_f64, 6_f64, 9_f64, 12_f64, 15_f64, 18_f64, 21_f64, 24_f64, 27_f64, 30_f64,
];
*/

const BETA: [f64; 7] = [1_f64, 3_f64, 6_f64, 18_f64, 24_f64, 27_f64, 30_f64];
//const BETA: [f64; 2] = [24_f64, 27_f64];

const DT: f64 = 0.000_01_f64; // prod

const FFT_RESOLUTION_SIZE: f64 = 0.1_f64;
static NUMBER_OF_MEASUREMENT: Lazy<usize> = Lazy::new(|| {
    ((1_f64 / DT) * 2_f64 / FFT_RESOLUTION_SIZE * (2_f64 * std::f64::consts::PI)).ceil() as usize
});

const LATTICE_DIM: usize = 16;

fn main() {
    let lean_mean = LatticeCyclique::<3>::new(1_f64, LATTICE_DIM)
        .unwrap()
        .get_number_of_points() as f64;

    std::fs::create_dir_all(format!("{}/{}", DIRECTORY, PLOT_SUB_DIR)).unwrap();
    const FIELD_NAMES: [&str; 2] = ["e", "b"];

    let mut result_e = Vec::with_capacity(11);
    let mut result_b = Vec::with_capacity(11);
    for (result, name) in [&mut result_e, &mut result_b]
        .iter_mut()
        .zip(FIELD_NAMES.iter())
    {
        for beta in &BETA {
            let file_name = format!("{}/mean_measures_corr_{}_{}.csv", DIRECTORY, name, beta);
            println!("reading {}", file_name);
            let read = read_file_csv::<[f64; 2]>(&file_name, *NUMBER_OF_MEASUREMENT).unwrap();
            (*result).push(read);
            //println!("finished reading {}", file_name);
        }
    }

    println!("all file loaded\nprocessing data\ninitialize FFT scratch");

    let mut results_fft_e = Vec::with_capacity(BETA.len());
    let mut results_fft_b = Vec::with_capacity(BETA.len());

    let len = result_e[0].len();
    let mut planner = FftPlanner::new();
    let fft_method = planner.plan_fft_forward(len);
    let mut scratch = vec![Complex::new(0_f64, 0_f64); fft_method.get_inplace_scratch_len()];

    for (index, beta) in BETA.iter().enumerate() {
        println!("processing data for beta = {}", beta);
        let mut res_fft_e = Vec::with_capacity(len / 2);
        let mut res_fft_b = Vec::with_capacity(len / 2);
        for (corr, fft_res, data_name) in &mut [
            (&mut result_e, &mut res_fft_e, "E"),
            (&mut result_b, &mut res_fft_b, "B"),
        ] {
            let (mut data_iter, mut variance_iter) = corr[index]
                .iter()
                .map(|[el, er]| (el, er))
                .unzip::<_, _, Vec<f64>, Vec<f64>>();
            let data = data_iter
                .drain(..)
                .map(|el| el.into())
                .collect::<Vec<Complex>>();
            let error = variance_iter
                .drain(..)
                .map(|var| (var * (lean_mean - 1_f64) / (lean_mean)).sqrt().into())
                .collect::<Vec<Complex>>();
            let mut data_fft = data;
            fft_method.process_with_scratch(&mut data_fft, &mut scratch);
            let mut error_fft = error;
            fft_method.process_with_scratch(&mut error_fft, &mut scratch);

            let normalization_fact = (data_fft.len() as f64).sqrt();
            let normalize = |input: &Complex| -> f64 {
                //real fft
                // why * 2.sqrt()?
                (2_f64 * input.real() * 2_f64.sqrt()) / normalization_fact
                // modulus fft
                //(input.modulus() * 2_f64.sqrt()) / normalization_fact
                // Complex fft
                //(2_f64 * input.imaginary() * 2_f64.sqrt()) / normalization_fact
            };
            plaquette::plot_corr_e::plot_data_fft(
                &data_fft,
                DT,
                &format!(
                    "{}/{}/plot_fft_{}_{}_log_log.svg",
                    DIRECTORY, PLOT_SUB_DIR, data_name, beta
                ),
            )
            .unwrap();

            fft_res.extend(
                data_fft
                    .iter()
                    .zip(error_fft.iter())
                    .take(data_fft.len() / 2)
                    .map(|(val, err)| [normalize(val), normalize(err)]),
            );
        }

        let file_name = format!(
            "{}/{}/plot_fft_norm_e_b_{}.svg",
            DIRECTORY, PLOT_SUB_DIR, beta
        );
        println!("plot {}", file_name);
        plot_fft_eb_err_norm(&file_name, &res_fft_e, &res_fft_b).unwrap();

        plot_fft_one(
            &format!("{}/{}/plot_fft_e_{}.svg", DIRECTORY, PLOT_SUB_DIR, beta),
            &res_fft_e,
            "E",
        )
        .unwrap();
        plot_fft_one(
            &format!("{}/{}/plot_fft_b_{}.svg", DIRECTORY, PLOT_SUB_DIR, beta),
            &res_fft_b,
            "B",
        )
        .unwrap();

        results_fft_e.push(res_fft_e);
        results_fft_b.push(res_fft_b);
    }
    drop(scratch);

    let file_name = format!("{}/{}/plot_fft_all_e.svg", DIRECTORY, PLOT_SUB_DIR);
    println!("plot {}", file_name);
    plot_fft_all(&file_name, &results_fft_e[1..], &BETA[1..], "E").unwrap();
    let file_name = format!("{}/{}/plot_fft_all_e_2.svg", DIRECTORY, PLOT_SUB_DIR);
    println!("plot {}", file_name);
    plot_fft_all(&file_name, &results_fft_e[2..], &BETA[2..], "E").unwrap();
    let file_name = format!("{}/{}/plot_fft_all_e_3.svg", DIRECTORY, PLOT_SUB_DIR);
    println!("plot {}", file_name);
    plot_fft_all(&file_name, &results_fft_e[3..], &BETA[3..], "E").unwrap();

    let file_name = format!("{}/{}/plot_fft_all_b.svg", DIRECTORY, PLOT_SUB_DIR);
    println!("plot {}", file_name);
    plot_fft_all(&file_name, &results_fft_b, &BETA, "B").unwrap();
}

const SIZE: (u32, u32) = (1920, 1080);
const MAX_W: f64 = 4_f64;

fn get_step_and_max(len: usize) -> (f64, usize) {
    let step = 1_f64 / (DT * len as f64 * 2_f64) * (2_f64 * std::f64::consts::PI);
    let max_step = ((MAX_W / step).ceil() as usize + 1).min(len);
    (step, max_step)
}

fn get_max_min<'a>(data: &mut impl Iterator<Item = &'a [f64; 2]>) -> Option<(f64, f64)> {
    let fist_data = data.next()?;
    let mut y_min = fist_data[0] - fist_data[1].abs();
    let mut y_max = fist_data[0] + fist_data[1].abs();
    for [el, er] in data {
        y_min = y_min.min(el - er.abs());
        y_max = y_max.max(el + er.abs());
    }
    Some((y_min, y_max))
}

fn plot_fft_eb_err_norm(
    file_name: &str,
    data_e: &[[f64; 2]],
    data_b: &[[f64; 2]],
) -> Result<(), Box<dyn Error>> {
    let (step, max_step) = get_step_and_max(data_e.len());
    //println!("step size = {}, numer_of_steps = {}", step, max_step);

    // skip the fist element
    let (y_e_min, y_e_max) = get_max_min(&mut data_e.iter().take(max_step).skip(1)).unwrap();
    let (y_b_min, y_b_max) = get_max_min(&mut data_b.iter().take(max_step).skip(1)).unwrap();

    let root = SVGBackend::new(file_name, SIZE).into_drawing_area();
    root.fill(&WHITE)?;

    // TODO axis sytle, legende
    let mut chart = ChartBuilder::on(&root)
        .margin(4)
        .margin_bottom(4)
        .x_label_area_size(140)
        .y_label_area_size(140)
        .right_y_label_area_size(140)
        .build_cartesian_2d(0_f64..(MAX_W + step), y_e_min..y_e_max)?
        .set_secondary_coord(0_f64..(MAX_W + step), y_b_min..y_b_max);

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .set_tick_mark_size(LabelAreaPosition::Bottom, 30)
        .y_desc("E")
        .x_desc("w")
        .axis_desc_style(("sans-serif", 60))
        .label_style(("sans-serif", 60))
        .draw()?;

    chart
        .configure_secondary_axes()
        .axis_desc_style(("sans-serif", 60))
        .label_style(("sans-serif", 60))
        .y_desc("B")
        .draw()?;

    chart
        .draw_series(
            data_e
                .iter()
                .take(max_step)
                .enumerate()
                .skip(1)
                .step_by(1)
                .map(|(index, [el, err])| {
                    ErrorBar::new_vertical(
                        index as f64 * step,
                        el - err.abs(),
                        *el,
                        el + err.abs(),
                        BLUE.filled(),
                        12,
                    )
                }),
        )?
        .label("E")
        .legend(|(x, y)| ErrorBar::new_vertical(x, y - 20, y, y + 20, BLUE.filled(), 12));
    chart
        .draw_secondary_series(
            data_b
                .iter()
                .take(max_step)
                .enumerate()
                .skip(1)
                .step_by(1)
                .map(|(index, [el, err])| {
                    ErrorBar::new_vertical(
                        index as f64 * step,
                        el - err.abs(),
                        *el,
                        el + err.abs(),
                        RED.filled(),
                        12,
                    )
                }),
        )?
        .label("B")
        .legend(|(x, y)| ErrorBar::new_vertical(x, y - 20, y, y + 20, RED.filled(), 12));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 45))
        .border_style(BLACK.stroke_width(2))
        .background_style(WHITE.filled())
        .draw()?;
    Ok(())
}

fn plot_fft_all(
    file_name: &str,
    data: &[Vec<[f64; 2]>],
    betas: &[f64],
    data_set: &str,
) -> Result<(), Box<dyn Error>> {
    let (step, max_step) = get_step_and_max(data[0].len());

    let (mut y_min, mut y_max) = get_max_min(&mut data[0].iter().take(max_step).skip(1)).unwrap();
    for vec in data.iter().skip(1) {
        let (y_e_min, y_e_max) = get_max_min(&mut vec.iter().take(max_step).skip(1)).unwrap();
        y_min = y_min.min(y_e_min);
        y_max = y_max.max(y_e_max);
    }

    let root = SVGBackend::new(file_name, SIZE).into_drawing_area();
    root.fill(&WHITE)?;

    // TODO axis sytle, legende
    let mut chart = ChartBuilder::on(&root)
        .margin(4)
        .margin_bottom(4)
        .x_label_area_size(140)
        .y_label_area_size(140)
        //.right_y_label_area_size(140)
        .build_cartesian_2d(0_f64..(MAX_W + step), y_min..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .set_tick_mark_size(LabelAreaPosition::Bottom, 30)
        .y_desc(data_set)
        .x_desc("w")
        .axis_desc_style(("sans-serif", 60))
        .label_style(("sans-serif", 60))
        .draw()?;

    let number_of_data = data.len();
    let color_step = 255 / (number_of_data as u8 - 1);
    for (data_index, (data_e, beta)) in data.iter().zip(betas.iter()).enumerate() {
        let color = RGBColor(
            255 - data_index as u8 * color_step,
            0,
            data_index as u8 * color_step,
        );
        chart
            .draw_series(
                data_e
                    .iter()
                    .take(max_step)
                    .enumerate()
                    .skip(1)
                    .step_by(1)
                    .map(|(index, [el, err])| {
                        ErrorBar::new_vertical(
                            index as f64 * step,
                            el - err.abs(),
                            *el,
                            el + err.abs(),
                            color.filled(),
                            12,
                        )
                    }),
            )?
            .label(format!("beta = {}  ", beta)) // two ending space for alligning correlcty
            .legend(move |(x, y)| ErrorBar::new_vertical(x, y - 20, y, y + 20, color.filled(), 12));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 45))
        .border_style(BLACK.stroke_width(2))
        .background_style(WHITE.filled())
        .draw()?;
    Ok(())
}

fn plot_fft_one(file_name: &str, data: &[[f64; 2]], data_set: &str) -> Result<(), Box<dyn Error>> {
    let (step, max_step) = get_step_and_max(data.len());

    let (y_min, y_max) = get_max_min(&mut data.iter().take(max_step).skip(1)).unwrap();

    let root = SVGBackend::new(file_name, SIZE).into_drawing_area();
    root.fill(&WHITE)?;

    // TODO axis sytle, legende
    let mut chart = ChartBuilder::on(&root)
        .margin(4)
        .margin_bottom(4)
        .x_label_area_size(140)
        .y_label_area_size(140)
        //.right_y_label_area_size(140)
        .build_cartesian_2d(0_f64..(MAX_W + step), y_min..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .set_tick_mark_size(LabelAreaPosition::Bottom, 30)
        .y_desc(data_set)
        .x_desc("w")
        .axis_desc_style(("sans-serif", 60))
        .label_style(("sans-serif", 60))
        .draw()?;

    chart.draw_series(
        data.iter()
            .take(max_step)
            .enumerate()
            .skip(1)
            .step_by(1)
            .map(|(index, [el, err])| {
                ErrorBar::new_vertical(
                    index as f64 * step,
                    el - err.abs(),
                    *el,
                    el + err.abs(),
                    BLACK.filled(),
                    12,
                )
            }),
    )?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 45))
        .border_style(BLACK.stroke_width(2))
        .background_style(WHITE.filled())
        .draw()?;
    Ok(())
}
