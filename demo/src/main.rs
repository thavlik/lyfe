//! Fluid simulation demo application.

mod app;
mod app_types;
mod cli;
mod colors;
mod events;
mod interaction;
mod lifecycle;
mod overlays;
mod rendering;
mod tooltip;
mod ui;

use anyhow::Result;
use app::DemoApp;
use clap::Parser;
use cli::Cli;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let scenario = cli.scenario_kind();

    log::info!("Starting Fluid Simulation Demo ({:?} scenario)", scenario);
    log::info!("Controls:");
    log::info!("  Mouse hover: Inspect cell");
    log::info!("  Space: Toggle pause");
    log::info!("  +/-: Adjust inspection mip");
    log::info!("  Escape: Exit");

    if cli.smoke_test {
        log::info!("Running in smoke-test mode (5 frames then exit)");
    }
    if cli.detail {
        log::info!("Detail mode enabled: inset play area with pinned inspectors");
    }
    log::info!("Present mode preference: {:?}", cli.present_mode);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = DemoApp::new(
        cli.smoke_test,
        scenario,
        cli.detail,
        cli.present_mode.into(),
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}
