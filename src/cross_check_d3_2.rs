use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use plaquette::{config_scan::*, data_analysis::*, observable, rng::*, sim::*};
use rayon::prelude::*;

fn main() {
    main_cross_check_volume();
}

pub fn get_values(min: f64, max: f64, pos: usize, number_of_pts: usize) -> f64 {
    min + (pos as f64) / (number_of_pts as f64 - 1_f64) * (max - min)
}

const N_ARRAY: [usize; 12] = [10, 11, 12, 13, 14, 15, 17, 18, 21, 24, 28, 34];

const BETA: f64 = 24_f64;

const SEED: u64 = 0x43_21_8d_93_a0_55_c3_5f;

fn main_cross_check_volume() {
    let beta = BETA;

    let vec_dim = N_ARRAY.to_vec();

    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64), //size
        ScanPossibility::Vector(vec_dim),   // dim
        ScanPossibility::Default(beta),     // Beta
    )
    .unwrap();
    let mc_cfg =
        MonteCarloConfigScan::new(ScanPossibility::Default(1), ScanPossibility::Default(0.1))
            .unwrap();
    let sim_cfg = SimConfigScan::new(
        mc_cfg,
        ScanPossibility::Default(10_000), //th steps (unused)
        ScanPossibility::Default(1),      // renormn (unused)
        ScanPossibility::Default(2_000),  // number_of_averages
        ScanPossibility::Default(500),    //between av
    )
    .unwrap();
    let config = ConfigScan::new(cfg_l, sim_cfg).unwrap();
    //println!("{:}", serde_json::to_string_pretty( &config).unwrap());
    let array_config = config.get_all_config();

    let multi_pb = std::sync::Arc::new(MultiProgress::new());
    let pb = multi_pb.add(ProgressBar::new(array_config.len() as u64));
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

    let _result_all_sim = array_config
        .par_iter()
        .enumerate()
        .map(|(index, cfg)| {
            let mut rng = get_rand_from_seed(SEED);
            for _ in 0..index {
                rng.jump();
            }
            let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
            let mut mc = get_mc_from_config_sweep(cfg.sim_config().mc_config(), rng);
            let (sim_th, t_exp) = thermalize_state(
                sim_init,
                &mut mc,
                &multi_pb,
                &observable::volume_obs,
                "",
                "d3",
            )
            .unwrap();
            //let (av, sim_final, _) = run_simulation_with_progress_bar_volume(cfg.sim_config(), sim_th, &multi_pb, rng);
            let _ = save_data_n(cfg, &sim_th, &"_th");

            let (sim_final, result) = simulation_gather_measurement(
                sim_th,
                &mut mc,
                &multi_pb,
                &observable::volume_obs,
                cfg.sim_config().number_of_steps_between_average(),
                cfg.sim_config().number_of_averages(),
            )
            .unwrap();
            let _ = write_vec_to_file_csv(
                &result,
                &format!(
                    "raw_measures_{}.csv",
                    cfg.lattice_config().lattice_number_of_points()
                ),
            );
            let _ = save_data_n(cfg, &sim_final, &"");
            pb.inc(1);
            (*cfg, t_exp)
        })
        .collect::<Vec<_>>();

    pb.finish();

    //let _ = write_data_to_file_csv_with_n(&result);
    //let _ = plot_data_volume(&result);
    let _ = h.join();
}
