use std::str::FromStr;

use clap::{Parser, Subcommand};
use renderer::PresentModePreference;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum ScenarioKind {
    #[default]
    Basic,
    AcidBase,
    Buffers,
    Catalyst,
    Enzyme,
    Leak,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum PresentModeCli {
    #[default]
    Auto,
    Fifo,
    Mailbox,
}

impl FromStr for PresentModeCli {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fifo" | "vsync" => Ok(Self::Fifo),
            "mailbox" | "low-latency" | "low_latency" => Ok(Self::Mailbox),
            _ => Err(format!(
                "unsupported present mode '{value}' (expected: auto, fifo, mailbox)"
            )),
        }
    }
}

impl From<PresentModeCli> for PresentModePreference {
    fn from(value: PresentModeCli) -> Self {
        match value {
            PresentModeCli::Auto => Self::Auto,
            PresentModeCli::Fifo => Self::Fifo,
            PresentModeCli::Mailbox => Self::Mailbox,
        }
    }
}

#[derive(Parser)]
#[command(name = "lyfe-demo", about = "Fluid simulation demo")]
pub(crate) struct Cli {
    #[arg(long)]
    pub(crate) smoke_test: bool,

    #[arg(long)]
    pub(crate) detail: bool,

    #[arg(long, default_value = "auto")]
    pub(crate) present_mode: PresentModeCli,

    #[command(subcommand)]
    command: Option<Commands>,
}

impl Cli {
    pub(crate) fn scenario_kind(&self) -> ScenarioKind {
        match self.command {
            Some(Commands::AcidBase) => ScenarioKind::AcidBase,
            Some(Commands::Buffers) => ScenarioKind::Buffers,
            Some(Commands::Catalyst) => ScenarioKind::Catalyst,
            Some(Commands::Enzyme) => ScenarioKind::Enzyme,
            Some(Commands::Leak) => ScenarioKind::Leak,
            Some(Commands::Basic) | None => ScenarioKind::Basic,
        }
    }
}

#[derive(Debug, Clone, Copy, Subcommand)]
enum Commands {
    Basic,
    AcidBase,
    Buffers,
    Catalyst,
    Enzyme,
    Leak,
}
