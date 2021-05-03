//! # Plaquette
//!
//! ![](https://img.shields.io/badge/language-Rust-orange)
//! ![Build](https://img.shields.io/github/workflow/status/ABouttefeux/plaquette/Rust)
//!
//! Simulation binary using  [lattice-qcd-rs](https://github.com/ABouttefeux/lattice-qcd-rs) that I use for my research.

//#![warn(clippy::as_conversions)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
//#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::implicit_hasher)]
#![warn(clippy::implicit_saturating_sub)]
#![warn(clippy::imprecise_flops)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::macro_use_imports)]
#![warn(clippy::manual_ok_or)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::non_ascii_literal)]
//#![warn(clippy::semicolon_if_nothing_returned)]
#![warn(clippy::suboptimal_flops)]
#![warn(clippy::todo)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::type_repetition_in_bounds)]
#![warn(clippy::unreadable_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::unused_self)]
#![warn(clippy::unnecessary_wraps)]
//#![warn(clippy::missing_errors_doc)]
//#![warn(missing_docs)]
#![forbid(unsafe_code)]

extern crate nalgebra as na;

pub mod config;
pub mod config_scan;
pub mod data_analysis;
pub mod io;
pub mod observable;
pub mod plot;
pub mod plot_corr_e;
pub mod rng;
pub mod sim;
#[cfg(test)]
mod test;
