use std::fs;

//use plotter_backend_text::*;
//use plotters::prelude::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lattice_qcd_rs::{
    error::StateInitializationError, integrator::SymplecticEulerRayon, simulation::*, statistics,
};
use nalgebra::Complex;
use once_cell::sync::Lazy;
use plaquette::{config::*, data_analysis::*, observable, rng::*, sim::*};
use rayon::prelude::*;

fn main() {
    main_sim();
}

const DIRECTORY: &str = &"data/data_set_wilson_loop/";

const BETA: f64 = 16_f64;
//const CF: f64 = 1.333_333_333_333_333_3_f64; // = (Nc^2-1)/(2 * Nc) = 4/3
//const DT: f64 = 0.000_1_f64; // test
const DT: f64 = 0.000_01_f64; // prod

const MAXT_IME: f64 = 50_f64;
static NUMBER_OF_MEASUREMENT: Lazy<usize> =
    Lazy::new(|| (MAXT_IME / DT / LATTICE_SIZE).ceil() as usize);

const SPACING_LEN: usize = 4;
const SPACING: [usize; SPACING_LEN] = [1, 2, 3, 4];

const LATTICE_DIM: usize = 12;
//const LATTICE_DIM: usize = 24;
//const LATTICE_DIM: usize = 8;
const LATTICE_SIZE: f64 = 1_f64;

const INTEGRATOR: SymplecticEulerRayon = SymplecticEulerRayon::new();
const SEED: u64 = 0x06_60_e9_e5_f5_27_9a_7c;

fn main_sim() {
    fs::create_dir_all(DIRECTORY).unwrap();
    let cfg_l = LatticeConfig::new(
        LATTICE_SIZE, //size
        LATTICE_DIM,  // dim
        BETA,         // Beta
    )
    .unwrap();
    let mc_cfg = MonteCarloConfig::new(1, 0.1).unwrap();
    let sim_cfg = SimConfig::new(
        mc_cfg, 10_000, //th steps (unused)
        1,      // renormn (unused)
        1_000,  // number_of_averages (unused)
        200,    //between av (unused)
    )
    .unwrap();
    let cfg = Config::new(cfg_l, sim_cfg);

    let multi_pb = std::sync::Arc::new(MultiProgress::new());
    let pb = multi_pb.add(ProgressBar::new(1));
    pb.set_style(
        ProgressStyle::default_bar()
            .progress_chars("=>-")
            .template(get_pb_template()),
    );
    pb.set_prefix("Sim Total");
    pb.enable_steady_tick(499);

    let multi_pb_2 = multi_pb.clone();
    let h = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1000));
        multi_pb_2.set_move_cursor(true);
        multi_pb_2.join_and_clear()
    });
    pb.tick();

    let mut rng = get_rand_from_seed(SEED);
    let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
    let mut mc = get_mc_from_config_sweep(cfg.sim_config().mc_config(), rng);
    /*
    let mut hb = HeatBathSweep::new(rng);
    let mut or1 = OverrelaxationSweepReverse::new();
    let mut or2 = OverrelaxationSweepReverse::new();
    let mut hm = HybrideMethode::new_empty();
    hm.push_methods(&mut hb);
    hm.push_methods(&mut or1);
    hm.push_methods(&mut or2);
    */

    let (sim_th, _t_exp) = thermalize_state(
        sim_init,
        &mut mc,
        &multi_pb,
        &observable::volume_obs,
        DIRECTORY,
        &format!("ecorr_{}", BETA),
    )
    .unwrap();
    let (state, _rng) = thermalize_with_e_field(sim_th, &multi_pb, mc.rng_owned(), DT).unwrap();

    // we may want to do one simulation step to remove the big jump at the begining
    //let state = state.simulate_symplectic(&INTEGRATOR, DT).unwrap();

    let _ = save_data_any(&state, &format!("{}/sim_bin_{}_th_e.bin", DIRECTORY, BETA));
    let (state, res) = measure(state, *NUMBER_OF_MEASUREMENT, &multi_pb).unwrap();

    let _ = save_data_any(&state, &format!("{}/sim_bin_{}_e.bin", DIRECTORY, BETA));
    let _ = write_vec_to_file_csv(
        &res,
        &format!("{}/mean_measures_wl_{}.csv", DIRECTORY, BETA),
    );
    let _ = write_vec_to_file_csv(
        &[SPACING],
        &format!("{}/mean_measures_wl_{}_info.csv", DIRECTORY, BETA),
    );

    pb.inc(1);

    pb.finish();
    let _ = h.join();
}

type ResultMeasure = (
    LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3>,
    Vec<[[Complex<f64>; 2]; SPACING_LEN]>,
);

#[allow(clippy::useless_format)]
fn measure(
    state_initial: LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3>,
    number_of_measurement: usize,
    mp: &MultiProgress,
) -> Result<ResultMeasure, StateInitializationError> {
    let pb = mp.add(ProgressBar::new((number_of_measurement) as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .progress_chars("=>-")
            .template(get_pb_template()),
    );
    pb.set_prefix(format!("simulating"));

    let mut state = state_initial.clone();
    let links = state.lattice().get_links().collect::<Vec<_>>();

    let mut vec_res = Vec::with_capacity(number_of_measurement + 1);

    {
        let mut array_data = [[Complex::from(0_f64); 2]; 4];
        for (index, data) in array_data.iter_mut().enumerate() {
            *data = statistics::mean_and_variance_par_iter_val(links.par_iter().map(|link| {
                observable::classical_wilson_loop(
                    &state_initial,
                    &state_initial,
                    *link.pos(),
                    link.dir(),
                    SPACING[index],
                )
                .unwrap()
            }));
        }
        vec_res.push(array_data);
    }
    //let mut vec_plot = vec![];
    //let mut y_min = 0_f64;
    //let mut y_max = 0_f64;

    for i in 0..number_of_measurement {
        let mut state_new = state.simulate_symplectic(&INTEGRATOR, DT)?;
        if i % 200 == 0 {
            pb.set_message(format!(
                "H {:.6} - G {:.6} ",
                state_new.hamiltonian_total(),
                state_new
                    .e_field()
                    .gauss_sum_div(state_new.link_matrix(), state_new.lattice())
                    .unwrap(),
            ));
            state_new.lattice_state_mut().normalize_link_matrices();

            /*
            let new_e = state_new.e_field().project_to_gauss(state_new.link_matrix(), state_new.lattice())
                .ok_or(SimulationError::NotValide)?;
            // let statement because of mutable_borrow_reservation_conflict
            // (https://github.com/rust-lang/rust/issues/59159)
            state_new.set_e_field(new_e);
            */
        }

        let mut array_data = [[Complex::from(0_f64); 2]; 4];
        for (index, data) in array_data.iter_mut().enumerate() {
            *data = statistics::mean_and_variance_par_iter_val(links.par_iter().map(|link| {
                observable::classical_wilson_loop(
                    &state_initial,
                    &state_new,
                    *link.pos(),
                    link.dir(),
                    SPACING[index],
                )
                .unwrap()
            }));
        }
        vec_res.push(array_data);
        /*
        const PLOT_COUNT: usize = 1_000;
        if i % PLOT_COUNT == 0 {
            // TODO clean, move to function
            let last_data = statistics::mean(vec.last().unwrap());
            vec_plot.push(last_data);
            if vec_plot.len() > 1 {
                y_min = y_min.min(last_data);
                y_max = y_max.max(last_data);
                if y_min < y_max {
                    let _ = draw_chart(
                        &TextDrawingBackend(vec![PixelState::Empty; 5000]).into_drawing_area(),
                        0_f64..((vec_plot.len() - 1) * PLOT_COUNT) as f64 * DT,
                        y_min..y_max,
                        vec_plot.iter().enumerate().map(|(index, el)| ((index * PLOT_COUNT) as f64 * DT, *el)),
                        "B Corr"
                    );
                    let _ = console::Term::stderr().move_cursor_up(30);
                }
            }
            else {
                y_min = last_data;
                y_max = last_data;
            }
        }
        */

        state = state_new;
        pb.inc(1);
    }

    pb.finish_and_clear();
    let _ = console::Term::stderr().move_cursor_down(30);
    Ok((state, vec_res))
}
