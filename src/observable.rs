use lattice_qcd_rs::{
    lattice::{Direction, DirectionList, LatticeLink, LatticePoint},
    simulation::*,
    CMatrix3, ComplexField,
};
use rayon::prelude::*;

pub fn volume_obs(p: &LatticePoint<3>, state: &LatticeStateDefault<3>) -> f64 {
    let number_of_directions = (Direction::<3>::dim() * (Direction::<3>::dim() - 1)) * 2; // ( *4 / 2)
    let directions_all = Direction::<3>::directions();
    // We consider all plaquette in positive and negative directions
    // but we avoid counting two times the plaquette P_IJ P_JI
    // as this is manage by taking the real part
    directions_all
        .iter()
        .map(|dir_1| {
            directions_all
                .iter()
                .filter(|dir_2| dir_1.index() > dir_2.index())
                .map(|dir_2| {
                    state
                        .link_matrix()
                        .pij(&p, &dir_1, &dir_2, state.lattice())
                        .map(|el| 1_f64 - el.trace().real() / 3_f64)
                        .unwrap()
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / number_of_directions as f64
}

pub fn volume_obs_mean(state: &LatticeStateDefault<3>) -> f64 {
    let sum = state
        .lattice()
        .get_points()
        .par_bridge()
        .map(|point| volume_obs(&point, state))
        .sum::<f64>();
    let number_of_plaquette = state.lattice().number_of_points() as f64;

    parameter_volume(sum / number_of_plaquette, state.beta())
}

#[allow(clippy::suboptimal_flops)] // readability
pub fn parameter_volume(value: f64, beta: f64) -> f64 {
    let c1: f64 = 8_f64 / 3_f64;
    const C2: f64 = 1.951_315_f64;
    const C3: f64 = 6.861_2_f64;
    const C4: f64 = 2.929_421_32_f64;
    beta.powi(4)
        * (value
            - c1 / beta
            - C2 / beta.powi(2)
            - C3 / beta.powi(3)
            - C4 * beta.ln() / beta.powi(4))
}

pub fn e_correletor(
    state: &LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3>,
    state_new: &LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3>,
    pt: &LatticePoint<3>,
) -> Option<f64> {
    Some(
        state_new
            .e_field()
            .e_vec(pt, state_new.lattice())?
            .iter()
            .zip(state.e_field().e_vec(pt, state.lattice())?.iter())
            .map(|(el1, el2)| {
                el1.iter()
                    .zip(el2.iter())
                    .map(|(d1, d2)| (d1 * d2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / (2_f64 * 3_f64),
    )
}

pub fn b_correletor(
    state: &LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3>,
    state_new: &LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3>,
    pt: &LatticePoint<3>,
) -> Option<f64> {
    Some(
        state_new
            .link_matrix()
            .magnetic_field_vec(pt, state_new.lattice())?
            .iter()
            .zip(
                state
                    .link_matrix()
                    .magnetic_field_vec(pt, state.lattice())?
                    .iter(),
            )
            .map(|(el1, el2)| (el1 * el2).trace().real())
            .sum::<f64>()
            / (3_f64),
    )
}

fn classical_wilson_loop_matrix<State, const D: usize>(
    state_zero: &State,
    state_new: &State,
    pt: LatticePoint<D>,
    dir: &Direction<D>,
    spacing: usize,
) -> Option<CMatrix3>
where
    State: LatticeState<D>,
{
    let mut pt = pt;
    let mut left_product = CMatrix3::identity();
    let mut right_product = CMatrix3::identity();
    for _ in 0..spacing {
        left_product *= state_zero
            .link_matrix()
            .matrix(&LatticeLink::new(pt, *dir), state_zero.lattice())?;
        right_product *= state_new
            .link_matrix()
            .matrix(&LatticeLink::new(pt, -dir), state_zero.lattice())?;
        pt = state_zero.lattice().add_point_direction(pt, dir);
    }
    Some(left_product * right_product.adjoint())
}

pub fn classical_wilson_loop<State, const D: usize>(
    state_zero: &State,
    state_new: &State,
    pt: LatticePoint<D>,
    dir: &Direction<D>,
    spacing: usize,
) -> Option<nalgebra::Complex<f64>>
where
    State: LatticeState<D>,
{
    Some(classical_wilson_loop_matrix(state_zero, state_new, pt, dir, spacing)?.trace())
}
