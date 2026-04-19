//! Vulkan renderer for the fluid simulation.
//!
//! This library provides visualization of the simulation state using Vulkan.
//! It renders:
//! - Fluid concentration fields as a 2D image
//! - Solid geometry with distinct coloring
//! - UI overlays for inspection tooltips
//!
//! ## Architecture
//!
//! The renderer consumes `RenderState` from fluidsim and produces frames.
//! It uses a simple fullscreen quad approach with a fragment shader that
//! samples concentration data and produces colors.

pub mod context;
pub mod egui_integration;
pub mod pipeline;

pub use context::{PresentModePreference, RenderContext};
pub use egui_integration::EguiRenderer;
pub use pipeline::{RenderPipeline, RenderViewport};
