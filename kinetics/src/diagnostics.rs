//! Diagnostic types for kinetics evaluation.
//!
//! These types are used to report warnings, errors, and informational
//! messages from the kinetics evaluation process.

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticLevel {
    /// Informational message (no action needed)
    Info,
    /// Warning (may indicate a problem)
    Warning,
    /// Error (something is wrong)
    Error,
    /// Debug (verbose debugging info)
    Debug,
}

impl DiagnosticLevel {
    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            DiagnosticLevel::Info => "INFO",
            DiagnosticLevel::Warning => "WARN",
            DiagnosticLevel::Error => "ERROR",
            DiagnosticLevel::Debug => "DEBUG",
        }
    }
}

/// A diagnostic message from kinetics evaluation.
#[derive(Debug, Clone)]
pub struct KineticsDiagnostic {
    /// Severity level
    pub level: DiagnosticLevel,
    /// The diagnostic message
    pub message: String,
    /// Optional tile ID this diagnostic relates to
    pub tile_id: Option<u32>,
    /// Optional boundary ID this diagnostic relates to
    pub boundary_id: Option<u32>,
    /// Optional code/category for the diagnostic
    pub code: Option<String>,
    /// Timestamp when this diagnostic was created
    pub timestamp_seconds: f64,
}

impl KineticsDiagnostic {
    /// Create a new diagnostic.
    pub fn new(level: DiagnosticLevel, message: String) -> Self {
        Self {
            level,
            message,
            tile_id: None,
            boundary_id: None,
            code: None,
            timestamp_seconds: 0.0,
        }
    }

    /// Create an info diagnostic.
    pub fn info(message: String) -> Self {
        Self::new(DiagnosticLevel::Info, message)
    }

    /// Create a warning diagnostic.
    pub fn warning(message: String) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    /// Create an error diagnostic.
    pub fn error(message: String) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    /// Create a debug diagnostic.
    pub fn debug(message: String) -> Self {
        Self::new(DiagnosticLevel::Debug, message)
    }

    /// Associate this diagnostic with a tile.
    pub fn with_tile(mut self, tile_id: u32) -> Self {
        self.tile_id = Some(tile_id);
        self
    }

    /// Associate this diagnostic with a boundary.
    pub fn with_boundary(mut self, boundary_id: u32) -> Self {
        self.boundary_id = Some(boundary_id);
        self
    }

    /// Set a diagnostic code.
    pub fn with_code(mut self, code: &str) -> Self {
        self.code = Some(code.to_string());
        self
    }

    /// Set the timestamp.
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp_seconds = timestamp;
        self
    }

    /// Check if this is an error.
    pub fn is_error(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Error)
    }

    /// Check if this is a warning or error.
    pub fn is_warning_or_error(&self) -> bool {
        matches!(
            self.level,
            DiagnosticLevel::Warning | DiagnosticLevel::Error
        )
    }
}

impl std::fmt::Display for KineticsDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.level.as_str(), self.message)?;
        if let Some(tile_id) = self.tile_id {
            write!(f, " (tile {})", tile_id)?;
        }
        if let Some(boundary_id) = self.boundary_id {
            write!(f, " (boundary {})", boundary_id)?;
        }
        if let Some(code) = &self.code {
            write!(f, " [{}]", code)?;
        }
        Ok(())
    }
}

/// A collection of diagnostics with utility methods.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticCollection {
    diagnostics: Vec<KineticsDiagnostic>,
}

impl DiagnosticCollection {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Add a diagnostic.
    pub fn push(&mut self, diagnostic: KineticsDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Add an info diagnostic.
    pub fn info(&mut self, message: String) {
        self.push(KineticsDiagnostic::info(message));
    }

    /// Add a warning diagnostic.
    pub fn warning(&mut self, message: String) {
        self.push(KineticsDiagnostic::warning(message));
    }

    /// Add an error diagnostic.
    pub fn error(&mut self, message: String) {
        self.push(KineticsDiagnostic::error(message));
    }

    /// Get all diagnostics.
    pub fn all(&self) -> &[KineticsDiagnostic] {
        &self.diagnostics
    }

    /// Get only errors.
    pub fn errors(&self) -> impl Iterator<Item = &KineticsDiagnostic> {
        self.diagnostics.iter().filter(|d| d.is_error())
    }

    /// Get warnings and errors.
    pub fn warnings_and_errors(&self) -> impl Iterator<Item = &KineticsDiagnostic> {
        self.diagnostics.iter().filter(|d| d.is_warning_or_error())
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    /// Get the count of diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Convert to a vec.
    pub fn into_vec(self) -> Vec<KineticsDiagnostic> {
        self.diagnostics
    }
}

impl IntoIterator for DiagnosticCollection {
    type Item = KineticsDiagnostic;
    type IntoIter = std::vec::IntoIter<KineticsDiagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.diagnostics.into_iter()
    }
}

impl<'a> IntoIterator for &'a DiagnosticCollection {
    type Item = &'a KineticsDiagnostic;
    type IntoIter = std::slice::Iter<'a, KineticsDiagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.diagnostics.iter()
    }
}
