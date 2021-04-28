//use na::{Complex, ComplexField};
use plotters::prelude::*;

use super::PlotType;

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn plot_data_fft(
    data: &[f64],
    delta_t: f64,
    file_name: &str,
    plot_type: PlotType,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut y_min = data[0];
    let mut y_max = data[0];
    for el in data {
        y_min = y_min.min(*el);
        y_max = y_max.max(*el);
    }

    let root = SVGBackend::new(file_name, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    const MAX_W: f64 = 4_f64;

    let step = 1_f64 / (delta_t * data.len() as f64 * 2_f64) * (2_f64 * std::f64::consts::PI);
    let max_step = ((MAX_W / step).ceil() as usize + 1).min(data.len());

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(120)
        .right_y_label_area_size(120)
        .build_cartesian_2d(0_f64..MAX_W, y_min..y_max)?;

    chart
        .configure_mesh()
        .y_desc("")
        .x_desc("w")
        .axis_desc_style(("sans-serif", 30))
        .label_style(("sans-serif", 30))
        .draw()?;

    match plot_type {
        PlotType::Line => {
            chart.draw_series(LineSeries::new(
                data.iter()
                    .take(max_step)
                    .enumerate()
                    .step_by(1)
                    .map(|(index, el)| (index as f64 * step, *el)),
                &BLACK,
            ))?;
        }
        PlotType::Circle => {
            chart.draw_series(
                data.iter()
                    .take(max_step)
                    .enumerate()
                    .step_by(1)
                    .map(|(index, el)| Circle::new((index as f64 * step, *el), 8, BLACK.filled())),
            )?;
        }
    }

    Ok(())
}
