use crate::species::SpeciesId;

#[derive(Debug, Clone)]
pub struct LeakChannel {
    pub rate: f32,
    pub species: SpeciesId,
    pub x: i32,
    pub y: i32,
    /// Stored as a raw 8-bit turn value via two's-complement reinterpretation.
    /// `0` = 0 degrees, `-1` = 255/256 of a full turn.
    pub rotation: i8,
}

impl LeakChannel {
    pub const EIGHTH_TURN: u8 = 32;

    pub fn new(rate: f32, species: SpeciesId, x: i32, y: i32, rotation: i8) -> Self {
        Self {
            rate,
            species,
            x,
            y,
            rotation,
        }
    }

    pub fn rotation_byte(&self) -> u8 {
        self.rotation as u8
    }

    pub fn set_rotation_byte(&mut self, rotation_byte: u8) {
        self.rotation = rotation_byte as i8;
    }

    pub fn rotate_eighth_turn(&mut self) {
        let next = self.rotation_byte().wrapping_add(Self::EIGHTH_TURN);
        self.set_rotation_byte(next);
    }

    pub fn angle_radians(&self) -> f32 {
        self.rotation_byte() as f32 * std::f32::consts::TAU / 256.0
    }

    pub fn transport_direction(&self) -> (f32, f32) {
        let angle = self.angle_radians();
        (angle.cos(), angle.sin())
    }

    pub fn cell_offset(&self) -> (i32, i32) {
        let (dir_x, dir_y) = self.transport_direction();
        let mut offset_x = dir_x.round() as i32;
        let offset_y = dir_y.round() as i32;

        if offset_x == 0 && offset_y == 0 {
            offset_x = 1;
        }

        (offset_x, offset_y)
    }

    pub fn sink_cell(&self) -> (i32, i32) {
        let (offset_x, offset_y) = self.cell_offset();
        (self.x - offset_x, self.y - offset_y)
    }

    pub fn source_cell(&self) -> (i32, i32) {
        let (offset_x, offset_y) = self.cell_offset();
        (self.x + offset_x, self.y + offset_y)
    }

    pub fn contains_grid_point(
        &self,
        grid_x: f32,
        grid_y: f32,
        half_length: f32,
        half_width: f32,
    ) -> bool {
        let dx = grid_x - self.x as f32;
        let dy = grid_y - self.y as f32;
        let angle = -self.angle_radians();
        let local_x = dx * angle.cos() - dy * angle.sin();
        let local_y = dx * angle.sin() + dy * angle.cos();
        local_x.abs() <= half_length && local_y.abs() <= half_width
    }

    pub fn flow_label(&self) -> &'static str {
        let (dir_x, dir_y) = self.transport_direction();
        if dir_x.abs() >= dir_y.abs() {
            if dir_x >= 0.0 {
                "left -> right"
            } else {
                "right -> left"
            }
        } else if dir_y >= 0.0 {
            "top -> bottom"
        } else {
            "bottom -> top"
        }
    }

    pub fn resolve_endpoints(
        &self,
        width: u32,
        height: u32,
        solid_mask: &[u32],
    ) -> Option<((i32, i32), (i32, i32))> {
        let (step_x, step_y) = self.cell_offset();
        let sink = self.find_first_fluid(width, height, solid_mask, -step_x, -step_y)?;
        let source = self.find_first_fluid(width, height, solid_mask, step_x, step_y)?;
        Some((sink, source))
    }

    fn find_first_fluid(
        &self,
        width: u32,
        height: u32,
        solid_mask: &[u32],
        step_x: i32,
        step_y: i32,
    ) -> Option<(i32, i32)> {
        let mut x = self.x + step_x;
        let mut y = self.y + step_y;
        let max_steps = width.max(height) as usize;

        for _ in 0..max_steps {
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                return None;
            }
            let index = (y as u32 * width + x as u32) as usize;
            if solid_mask.get(index).copied().unwrap_or(1) == 0 {
                return Some((x, y));
            }
            x += step_x;
            y += step_y;
        }

        None
    }
}