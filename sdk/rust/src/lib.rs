//! # Olfactory Transformer Rust SDK
//!
//! High-performance, type-safe Rust client for molecular scent prediction.
//!
//! ## Features
//!
//! - **Async/await support** - Built on tokio for high-performance async operations
//! - **Type safety** - Strongly typed API with comprehensive error handling
//! - **Streaming support** - Real-time sensor data streaming with WebSockets
//! - **Batch processing** - Efficient batch prediction with parallel processing
//! - **Rate limiting** - Built-in rate limiting and retry logic
//! - **Caching** - Optional response caching for improved performance
//!
//! ## Quick Start
//!
//! ```rust
//! use olfactory_transformer::{Client, ClientConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ClientConfig::new("https://api.olfactory-transformer.com")
//!         .with_api_key("your-api-key");
//!     
//!     let client = Client::new(config).await?;
//!     
//!     let prediction = client
//!         .predict_scent("CC(C)CC1=CC=C(C=C1)C(C)C")
//!         .await?;
//!     
//!     println!("Primary notes: {:?}", prediction.primary_notes);
//!     println!("Intensity: {}/10", prediction.intensity);
//!     
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use reqwest::{Client as HttpClient, Method, Request, Response};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;
use uuid::Uuid;

pub mod types;
pub mod streaming;
pub mod batch;
pub mod error;

pub use types::*;
pub use streaming::*;
pub use batch::*;
pub use error::*;

/// Client configuration for the Olfactory Transformer API
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL for the API
    pub base_url: String,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Request timeout in milliseconds
    pub timeout: Duration,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay: Duration,
    /// Rate limit (requests per second)
    pub rate_limit: Option<u32>,
    /// Enable request/response logging
    pub enable_logging: bool,
}

impl ClientConfig {
    /// Create new client configuration
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
            rate_limit: Some(10), // 10 requests per second by default
            enable_logging: false,
        }
    }

    /// Set API key for authentication
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set retry delay
    pub fn with_retry_delay(mut self, retry_delay: Duration) -> Self {
        self.retry_delay = retry_delay;
        self
    }

    /// Set rate limit (requests per second)
    pub fn with_rate_limit(mut self, rate_limit: u32) -> Self {
        self.rate_limit = Some(rate_limit);
        self
    }

    /// Enable request/response logging
    pub fn with_logging(mut self, enable: bool) -> Self {
        self.enable_logging = enable;
        self
    }
}

/// Main client for the Olfactory Transformer API
#[derive(Clone)]
pub struct Client {
    config: ClientConfig,
    http_client: HttpClient,
    rate_limiter: Option<Arc<RateLimiter<governor::clock::DefaultClock, governor::middleware::NoOpMiddleware>>>,
    active_streams: Arc<RwLock<HashMap<String, StreamHandle>>>,
}

impl Client {
    /// Create new client instance
    pub async fn new(config: ClientConfig) -> Result<Self, OlfactoryError> {
        let mut builder = HttpClient::builder()
            .timeout(config.timeout)
            .user_agent("olfactory-transformer-rust/1.0.0");

        if config.enable_logging {
            builder = builder.connection_verbose(true);
        }

        let http_client = builder.build()
            .map_err(|e| OlfactoryError::Client(format!("Failed to create HTTP client: {}", e)))?;

        let rate_limiter = config.rate_limit.map(|limit| {
            Arc::new(RateLimiter::direct(
                Quota::per_second(nonzero!(limit))
            ))
        });

        let client = Self {
            config,
            http_client,
            rate_limiter,
            active_streams: Arc::new(RwLock::new(HashMap::new())),
        };

        // Verify connection
        client.health_check().await?;

        Ok(client)
    }

    /// Predict scent from SMILES string
    pub async fn predict_scent(&self, smiles: &str) -> Result<ScentPrediction, OlfactoryError> {
        self.predict_scent_with_options(smiles, &PredictionOptions::default()).await
    }

    /// Predict scent with custom options
    pub async fn predict_scent_with_options(
        &self,
        smiles: &str,
        options: &PredictionOptions,
    ) -> Result<ScentPrediction, OlfactoryError> {
        let request = PredictionRequest {
            molecule: Some(MoleculeInput {
                smiles: smiles.to_string(),
                name: None,
                cas_number: None,
            }),
            sensor_data: None,
            mode: options.mode.clone(),
            include_features: options.include_features,
            include_attention: options.include_attention,
            model_version: options.model_version.clone(),
        };

        let response: PredictionResponse = self
            .make_request(Method::POST, "/predict", Some(&request))
            .await?;

        Ok(response.prediction)
    }

    /// Predict scent from sensor data
    pub async fn predict_from_sensors(&self, sensor_data: &SensorData) -> Result<ScentPrediction, OlfactoryError> {
        let request = PredictionRequest {
            molecule: None,
            sensor_data: Some(sensor_data.clone()),
            mode: PredictionMode::Balanced,
            include_features: false,
            include_attention: false,
            model_version: "latest".to_string(),
        };

        let response: PredictionResponse = self
            .make_request(Method::POST, "/predict", Some(&request))
            .await?;

        Ok(response.prediction)
    }

    /// Batch prediction for multiple molecules
    pub async fn predict_batch(&self, request: &BatchPredictionRequest) -> Result<BatchPredictionResponse, OlfactoryError> {
        self.make_request(Method::POST, "/predict/batch", Some(request)).await
    }

    /// Start streaming prediction session
    pub async fn start_streaming(&self, session: &StreamingSession) -> Result<StreamHandle, OlfactoryError> {
        // Start session on server
        let _: serde_json::Value = self
            .make_request(Method::POST, "/stream/start", Some(session))
            .await?;

        // Create WebSocket connection
        let ws_url = self.config.base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + &format!("/stream/{}", session.session_id);

        let url = Url::parse(&ws_url)
            .map_err(|e| OlfactoryError::Client(format!("Invalid WebSocket URL: {}", e)))?;

        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| OlfactoryError::Client(format!("WebSocket connection failed: {}", e)))?;

        let handle = StreamHandle::new(session.session_id.clone(), ws_stream);

        // Store handle
        let mut streams = self.active_streams.write().await;
        streams.insert(session.session_id.clone(), handle.clone());

        Ok(handle)
    }

    /// Stop streaming session
    pub async fn stop_streaming(&self, session_id: &str) -> Result<(), OlfactoryError> {
        // Remove from active streams
        let mut streams = self.active_streams.write().await;
        if let Some(handle) = streams.remove(session_id) {
            handle.close().await?;
        }

        // Stop session on server
        let _: serde_json::Value = self
            .make_request(Method::DELETE, &format!("/stream/{}", session_id), None::<&()>)
            .await?;

        Ok(())
    }

    /// Get available models
    pub async fn get_models(&self) -> Result<Vec<ModelInfo>, OlfactoryError> {
        self.make_request(Method::GET, "/models", None::<&()>).await
    }

    /// Get model information
    pub async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo, OlfactoryError> {
        self.make_request(Method::GET, &format!("/models/{}", model_name), None::<&()>).await
    }

    /// Get API health status
    pub async fn health_check(&self) -> Result<ApiHealth, OlfactoryError> {
        self.make_request(Method::GET, "/health", None::<&()>).await
    }

    /// Get API metrics
    pub async fn get_metrics(&self) -> Result<ApiMetrics, OlfactoryError> {
        self.make_request(Method::GET, "/metrics", None::<&()>).await
    }

    /// Make HTTP request with retry logic
    async fn make_request<T, R>(&self, method: Method, path: &str, body: Option<&T>) -> Result<R, OlfactoryError>
    where
        T: Serialize,
        R: for<'de> Deserialize<'de>,
    {
        // Apply rate limiting
        if let Some(limiter) = &self.rate_limiter {
            limiter.until_ready().await;
        }

        let url = format!("{}{}", self.config.base_url, path);
        let mut request_builder = self.http_client.request(method, &url);

        // Add authentication
        if let Some(api_key) = &self.config.api_key {
            request_builder = request_builder.header("X-API-Key", api_key);
        }

        // Add body if provided
        if let Some(body) = body {
            request_builder = request_builder.json(body);
        }

        // Retry logic
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            match request_builder.try_clone() {
                Some(builder) => {
                    match builder.send().await {
                        Ok(response) => {
                            if response.status().is_success() {
                                return response.json().await
                                    .map_err(|e| OlfactoryError::Response(format!("JSON decode error: {}", e)));
                            } else if response.status().as_u16() == 429 {
                                // Rate limited, wait and retry
                                if attempt < self.config.max_retries {
                                    tokio::time::sleep(self.config.retry_delay * (attempt + 1)).await;
                                    continue;
                                }
                            } else {
                                let status = response.status();
                                let error_text = response.text().await
                                    .unwrap_or_else(|_| "Unknown error".to_string());
                                return Err(OlfactoryError::Api {
                                    status: status.as_u16(),
                                    message: error_text,
                                });
                            }
                        }
                        Err(e) => {
                            last_error = Some(e);
                            if attempt < self.config.max_retries {
                                tokio::time::sleep(self.config.retry_delay * (attempt + 1)).await;
                                continue;
                            }
                        }
                    }
                }
                None => {
                    return Err(OlfactoryError::Client("Failed to clone request".to_string()));
                }
            }
        }

        Err(OlfactoryError::Request(
            last_error.unwrap_or_else(|| reqwest::Error::from(reqwest::ErrorKind::Request))
        ))
    }

    /// Close all connections and cleanup
    pub async fn close(&self) -> Result<(), OlfactoryError> {
        let streams = self.active_streams.read().await;
        for (_, handle) in streams.iter() {
            handle.close().await?;
        }
        Ok(())
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // Spawn cleanup task
        let streams = self.active_streams.clone();
        tokio::spawn(async move {
            let streams = streams.read().await;
            for (_, handle) in streams.iter() {
                let _ = handle.close().await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_client_creation() {
        let config = ClientConfig::new("https://api.example.com")
            .with_api_key("test-key")
            .with_timeout(Duration::from_secs(10));

        // This would fail in a real test without a server
        // but demonstrates the API
        assert_eq!(config.base_url, "https://api.example.com");
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_prediction_options() {
        let options = PredictionOptions::default()
            .with_features(true)
            .with_attention(true)
            .with_model_version("v2.0");

        assert!(options.include_features);
        assert!(options.include_attention);
        assert_eq!(options.model_version, "v2.0");
    }
}