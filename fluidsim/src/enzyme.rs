use glam::Vec2;

#[derive(Debug, Clone, Copy)]
pub struct EnzymeField {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub hot_temperature: f32,
    pub cool_temperature: f32,
    pub circulation_strength: f32,
    pub thermophoretic_strength: f32,
    pub brownian_strength: f32,
    pub rotational_diffusion: f32,
}

impl EnzymeField {
    pub fn reflect_position(&self, position: Vec2) -> Vec2 {
        Vec2::new(
            reflect_axis(position.x, self.min_x, self.max_x),
            reflect_axis(position.y, self.min_y, self.max_y),
        )
    }

    pub fn thermal_drift(&self, thermal_bias: f32) -> Vec2 {
        let delta_temperature = (self.hot_temperature - self.cool_temperature).abs();
        if delta_temperature <= f32::EPSILON {
            return Vec2::ZERO;
        }

        // Keep the thermal transport much weaker than Brownian diffusion.
        // With the current 15 K split this remains a small perturbation,
        // preserving an approximately uniform equilibrium distribution.
        let gradient_direction = Vec2::new(-1.0, -1.0).normalize();
        let drift_scale = self.thermophoretic_strength
            * self.circulation_strength
            * (delta_temperature / 300.0)
            * thermal_bias;
        gradient_direction * drift_scale
    }
}

#[derive(Debug, Clone)]
pub struct EnzymeEntity {
    pub x: f32,
    pub y: f32,
    pub rotation_radians: f32,
    pub catalytic_scale: f32,
    pub mobility_scale: f32,
    pub thermal_bias: f32,
    rng_state: u32,
}

impl EnzymeEntity {
    pub const BODY_HALF_WIDTH: f32 = 5.2;
    pub const BODY_HALF_HEIGHT: f32 = 3.6;
    pub const ACTIVE_SITE_OFFSET: Vec2 = Vec2::new(4.4, 0.0);

    pub fn new(
        x: f32,
        y: f32,
        rotation_radians: f32,
        catalytic_scale: f32,
        mobility_scale: f32,
        thermal_bias: f32,
        seed: u32,
    ) -> Self {
        Self {
            x,
            y,
            rotation_radians,
            catalytic_scale,
            mobility_scale,
            thermal_bias,
            rng_state: seed.max(1),
        }
    }

    pub fn position(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn active_site_position(&self) -> Vec2 {
        self.position() + rotate(self.rotation_radians, Self::ACTIVE_SITE_OFFSET)
    }

    pub fn active_site_cell(&self) -> (i32, i32) {
        let site = self.active_site_position();
        (site.x.round() as i32, site.y.round() as i32)
    }

    pub fn advance(&mut self, dt: f32, field: &EnzymeField) {
        let drift = field.thermal_drift(self.thermal_bias) * self.mobility_scale;
        let noise = Vec2::new(self.next_gaussian(), self.next_gaussian())
            * (field.brownian_strength * self.mobility_scale * dt.sqrt());
        let candidate = field.reflect_position(self.position() + drift * dt + noise);
        self.x = candidate.x;
        self.y = candidate.y;

        self.rotation_radians = wrap_angle(
            self.rotation_radians + self.next_gaussian() * field.rotational_diffusion * dt.sqrt(),
        );
    }

    fn next_uniform(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.rng_state >> 8) as f32 / ((u32::MAX >> 8) as f32)
    }

    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_uniform().clamp(1e-6, 1.0 - 1e-6);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn reflect_axis(value: f32, min: f32, max: f32) -> f32 {
    let span = (max - min).max(1e-6);
    let period = 2.0 * span;
    let wrapped = (value - min).rem_euclid(period);
    if wrapped <= span {
        min + wrapped
    } else {
        max - (wrapped - span)
    }
}

fn rotate(angle: f32, vector: Vec2) -> Vec2 {
    let (sin_theta, cos_theta) = angle.sin_cos();
    Vec2::new(
        vector.x * cos_theta - vector.y * sin_theta,
        vector.x * sin_theta + vector.y * cos_theta,
    )
}

fn wrap_angle(angle: f32) -> f32 {
    let mut wrapped = angle;
    while wrapped > std::f32::consts::PI {
        wrapped -= std::f32::consts::TAU;
    }
    while wrapped < -std::f32::consts::PI {
        wrapped += std::f32::consts::TAU;
    }
    wrapped
}
