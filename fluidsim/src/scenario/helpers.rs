use super::ScenarioBuilder;

#[derive(Debug, Clone, Copy)]
pub(super) struct CentralBoxBounds {
    pub wall_thickness: u32,
    pub outer_x0: u32,
    pub outer_y0: u32,
    pub outer_x1: u32,
    pub outer_y1: u32,
    pub inner_x0: u32,
    pub inner_y0: u32,
    pub inner_x1: u32,
    pub inner_y1: u32,
}

pub(super) fn central_box_bounds(width: u32, height: u32) -> CentralBoxBounds {
    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64);
    let outer_size = inner_size + 2 * wall_thickness;

    let center_x = width / 2;
    let center_y = height / 2;

    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;

    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    CentralBoxBounds {
        wall_thickness,
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    }
}

pub(super) fn add_titanium_hollow_box(
    builder: ScenarioBuilder,
    bounds: CentralBoxBounds,
) -> ScenarioBuilder {
    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);
    builder.fill_hollow_rect(
        bounds.outer_x0,
        bounds.outer_y0,
        bounds.outer_x1,
        bounds.outer_y1,
        bounds.wall_thickness,
        titanium,
    )
}

pub(super) fn normalized_position(bounds: CentralBoxBounds, x: u32, y: u32) -> (f32, f32) {
    let inner_width = (bounds.inner_x1 - bounds.inner_x0) as f32;
    let inner_height = (bounds.inner_y1 - bounds.inner_y0) as f32;
    let nx = (x - bounds.inner_x0) as f32 / inner_width;
    let ny = (y - bounds.inner_y0) as f32 / inner_height;
    (nx, ny)
}

pub(super) fn lcg_next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

pub(super) fn lcg_unit(seed: &mut u32) -> f32 {
    (lcg_next(seed) >> 8) as f32 / ((u32::MAX >> 8) as f32)
}
