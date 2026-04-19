//! No-op evaluator for explicitly disabled rules and testing.
//!
//! This evaluator returns no-op updates for all snapshots.
//! It is used when:
//! - Semantic rule evaluation is explicitly disabled
//! - Testing/debugging the integration

use crate::{
    KineticsConfig, KineticsError, SemanticSnapshot, SemanticUpdate, engine::RuleEvaluator,
};

/// A no-op rule evaluator that returns empty updates.
#[derive(Debug, Clone, Default)]
pub struct NoopEvaluator {
    /// Number of evaluations performed
    evaluation_count: u64,
}

impl NoopEvaluator {
    /// Create a new no-op evaluator.
    pub fn new() -> Self {
        Self {
            evaluation_count: 0,
        }
    }
}

impl RuleEvaluator for NoopEvaluator {
    fn name(&self) -> &str {
        "no-op"
    }

    fn evaluate(
        &mut self,
        snapshot: &SemanticSnapshot,
        config: &KineticsConfig,
    ) -> Result<SemanticUpdate, KineticsError> {
        self.evaluation_count += 1;

        if config.verbose {
            log::debug!(
                "NoopEvaluator: returning no-op update for snapshot at t={:.3}s ({} tiles)",
                snapshot.sim_time_seconds,
                snapshot.tile_count()
            );
        }

        // Return a no-op update valid for the configured interval
        let valid_from = snapshot.sim_time_seconds;
        let valid_until = valid_from + config.min_evaluation_interval;

        Ok(SemanticUpdate::noop(valid_from, valid_until))
    }

    fn is_available(&self) -> bool {
        true // Always available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_evaluator() {
        let mut evaluator = NoopEvaluator::new();
        let snapshot = SemanticSnapshot::new(8, 8, 64, 512, 512);
        let config = KineticsConfig::default();

        let update = evaluator.evaluate(&snapshot, &config).unwrap();

        assert!(update.is_noop());
        assert_eq!(update.valid_from_time_seconds, 0.0);
        assert!(update.valid_until_time_seconds > 0.0);
    }
}
