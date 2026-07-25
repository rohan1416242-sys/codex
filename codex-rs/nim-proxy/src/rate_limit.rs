//! Simple per-process rate limiter — sliding-window over the last 60 seconds.
//!
//! NIM's free tier is 40 requests/minute. We track timestamps of recent
//! requests and sleep before forwarding if we'd exceed the limit.

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use tokio::sync::Mutex;
use tracing::debug;
use tracing::info;
use tracing::warn;

/// Default rate limit (NIM free tier = 40 req/min).
pub const DEFAULT_RPM: u32 = 40;

#[derive(Debug, Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<Vec<Instant>>>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    pub fn new(max_requests_per_minute: u32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::with_capacity(
                max_requests_per_minute as usize + 1,
            ))),
            max_requests: max_requests_per_minute as usize,
            window: Duration::from_secs(60),
        }
    }

    pub async fn acquire(&self) {
        loop {
            let sleep_for = {
                let mut timestamps = self.inner.lock().await;
                let now = Instant::now();
                timestamps.retain(|t| now.duration_since(*t) < self.window);

                if timestamps.len() < self.max_requests {
                    timestamps.push(now);
                    debug!(
                        "rate_limiter: slot acquired ({}/{})",
                        timestamps.len(),
                        self.max_requests
                    );
                    return;
                }

                let oldest = timestamps[0];
                let elapsed = now.duration_since(oldest);
                let wait = self
                    .window
                    .checked_sub(elapsed)
                    .unwrap_or(Duration::from_millis(100))
                    + Duration::from_millis(50);
                warn!(
                    "rate_limiter: at {} rpm cap, sleeping {:?} before next request",
                    self.max_requests, wait
                );
                wait
            };

            tokio::time::sleep(sleep_for).await;
        }
    }

    pub async fn current_count(&self) -> usize {
        let mut timestamps = self.inner.lock().await;
        let now = Instant::now();
        timestamps.retain(|t| now.duration_since(*t) < self.window);
        timestamps.len()
    }
}

pub fn spawn_stats_logger(limiter: RateLimiter) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let count = limiter.current_count().await;
            info!("rate_limiter: {count} requests in the last 60s");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allows_up_to_max_requests_without_sleep() {
        let limiter = RateLimiter::new(3);
        let t0 = Instant::now();
        limiter.acquire().await;
        limiter.acquire().await;
        limiter.acquire().await;
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(50),
            "first 3 should not sleep, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn blocks_when_limit_reached() {
        let limiter = RateLimiter {
            inner: Arc::new(Mutex::new(Vec::new())),
            max_requests: 1,
            window: Duration::from_millis(200),
        };
        limiter.acquire().await;
        let t0 = Instant::now();
        limiter.acquire().await;
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(150),
            "second acquire should sleep, only took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn current_count_drops_old_entries() {
        let limiter = RateLimiter {
            inner: Arc::new(Mutex::new(Vec::new())),
            max_requests: 10,
            window: Duration::from_millis(50),
        };
        limiter.acquire().await;
        assert_eq!(limiter.current_count().await, 1);
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(limiter.current_count().await, 0);
    }
}
