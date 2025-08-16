/**
 * Olfactory Transformer JavaScript SDK
 * 
 * High-performance, type-safe client for molecular scent prediction
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import WebSocket from 'ws';
import { EventEmitter } from 'eventemitter3';

// Type definitions
export interface MoleculeInput {
  smiles: string;
  name?: string;
  casNumber?: string;
}

export interface SensorData {
  readings: Record<string, number>;
  timestamp?: Date;
  sensorTypes: string[];
  calibrationApplied?: boolean;
}

export interface ScentPrediction {
  primaryNotes: string[];
  descriptors: string[];
  intensity: number;
  confidence: number;
  chemicalFamily?: string;
  categories: string[];
}

export interface PredictionResponse {
  prediction: ScentPrediction;
  molecularFeatures?: MolecularFeatures;
  attentionWeights?: AttentionWeights;
  metadata: Record<string, any>;
  processingTimeMs: number;
  modelVersion: string;
  cached: boolean;
}

export interface MolecularFeatures {
  molecularWeight?: number;
  logp?: number;
  tpsa?: number;
  numAtoms?: number;
  numBonds?: number;
  numRings?: number;
  aromaticRings?: number;
  rotatableBonds?: number;
}

export interface AttentionWeights {
  layerWeights: number[][];
  headWeights: number[][];
  tokenWeights: number[];
}

export interface BatchPredictionRequest {
  molecules: MoleculeInput[];
  mode?: 'fast' | 'accurate' | 'balanced';
  includeFeatures?: boolean;
  parallelProcessing?: boolean;
}

export interface BatchPredictionResponse {
  results: BatchPredictionResult[];
  summary: Record<string, any>;
  totalProcessingTimeMs: number;
  batchId: string;
}

export interface BatchPredictionResult {
  molecule: MoleculeInput;
  prediction?: ScentPrediction;
  error?: string;
  success: boolean;
  processingTimeMs: number;
}

export interface StreamingSession {
  sessionId: string;
  sensorConfig: Record<string, any>;
  samplingRate?: number;
  bufferSize?: number;
  autoPrediction?: boolean;
}

export interface StreamingPrediction {
  sessionId: string;
  timestamp: Date;
  prediction: ScentPrediction;
  sensorData: SensorData;
  sequenceNumber: number;
}

export interface ClientConfig {
  baseURL: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  rateLimitBuffer?: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  status: string;
  parameters: string;
  architecture: string;
  capabilities: string[];
  trainingData: Record<string, any>;
  performanceMetrics: Record<string, number>;
  lastUpdated: Date;
}

/**
 * Main client class for Olfactory Transformer API
 */
export class OlfactoryClient extends EventEmitter {
  private client: AxiosInstance;
  private config: ClientConfig;
  private activeStreams: Map<string, WebSocket> = new Map();

  constructor(config: ClientConfig) {
    super();
    this.config = {
      timeout: 30000,
      retries: 3,
      retryDelay: 1000,
      rateLimitBuffer: 100,
      ...config
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { 'X-API-Key': this.config.apiKey })
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor for rate limiting
    this.client.interceptors.request.use(async (config) => {
      // Add rate limiting logic here
      await this.checkRateLimit();
      return config;
    });

    // Response interceptor for retry logic
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (this.shouldRetry(error)) {
          return this.retryRequest(error);
        }
        throw error;
      }
    );
  }

  private async checkRateLimit(): Promise<void> {
    // Implement rate limiting logic
    // For now, just a simple delay
    if (this.config.rateLimitBuffer) {
      await new Promise(resolve => setTimeout(resolve, this.config.rateLimitBuffer));
    }
  }

  private shouldRetry(error: any): boolean {
    return (
      error.response?.status >= 500 ||
      error.response?.status === 429 ||
      error.code === 'ECONNRESET' ||
      error.code === 'ETIMEDOUT'
    );
  }

  private async retryRequest(error: any, attempt: number = 1): Promise<any> {
    if (attempt > (this.config.retries || 3)) {
      throw error;
    }

    const delay = (this.config.retryDelay || 1000) * Math.pow(2, attempt - 1);
    await new Promise(resolve => setTimeout(resolve, delay));

    try {
      return await this.client.request(error.config);
    } catch (retryError) {
      return this.retryRequest(retryError, attempt + 1);
    }
  }

  /**
   * Predict scent from SMILES string
   */
  async predictScent(
    smiles: string,
    options: {
      includeFeatures?: boolean;
      includeAttention?: boolean;
      modelVersion?: string;
    } = {}
  ): Promise<PredictionResponse> {
    try {
      const response = await this.client.post('/predict', {
        molecule: { smiles },
        include_features: options.includeFeatures || false,
        include_attention: options.includeAttention || false,
        model_version: options.modelVersion || 'latest'
      });

      return this.transformResponse(response.data);
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Prediction failed: ${error}`);
    }
  }

  /**
   * Predict scent from sensor data
   */
  async predictFromSensors(sensorData: SensorData): Promise<PredictionResponse> {
    try {
      const response = await this.client.post('/predict', {
        sensor_data: {
          readings: sensorData.readings,
          timestamp: sensorData.timestamp?.toISOString(),
          sensor_types: sensorData.sensorTypes,
          calibration_applied: sensorData.calibrationApplied || false
        }
      });

      return this.transformResponse(response.data);
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Sensor prediction failed: ${error}`);
    }
  }

  /**
   * Batch prediction for multiple molecules
   */
  async predictBatch(request: BatchPredictionRequest): Promise<BatchPredictionResponse> {
    try {
      const response = await this.client.post('/predict/batch', {
        molecules: request.molecules,
        mode: request.mode || 'balanced',
        include_features: request.includeFeatures || false,
        parallel_processing: request.parallelProcessing !== false
      });

      return this.transformBatchResponse(response.data);
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Batch prediction failed: ${error}`);
    }
  }

  /**
   * Start streaming prediction session
   */
  async startStreaming(session: StreamingSession): Promise<void> {
    try {
      // Start session
      await this.client.post('/stream/start', session);

      // Create WebSocket connection
      const wsUrl = this.config.baseURL.replace('http', 'ws') + `/stream/${session.sessionId}`;
      const ws = new WebSocket(wsUrl);

      this.activeStreams.set(session.sessionId, ws);

      ws.on('open', () => {
        this.emit('streamConnected', session.sessionId);
      });

      ws.on('message', (data: Buffer) => {
        try {
          const prediction = JSON.parse(data.toString()) as StreamingPrediction;
          this.emit('streamingPrediction', prediction);
        } catch (error) {
          this.emit('streamError', { sessionId: session.sessionId, error });
        }
      });

      ws.on('error', (error) => {
        this.emit('streamError', { sessionId: session.sessionId, error });
      });

      ws.on('close', () => {
        this.activeStreams.delete(session.sessionId);
        this.emit('streamDisconnected', session.sessionId);
      });

    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to start streaming: ${error}`);
    }
  }

  /**
   * Stop streaming session
   */
  async stopStreaming(sessionId: string): Promise<void> {
    try {
      const ws = this.activeStreams.get(sessionId);
      if (ws) {
        ws.close();
        this.activeStreams.delete(sessionId);
      }

      await this.client.delete(`/stream/${sessionId}`);
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to stop streaming: ${error}`);
    }
  }

  /**
   * Get available models
   */
  async getModels(): Promise<ModelInfo[]> {
    try {
      const response = await this.client.get('/models');
      return response.data.map(this.transformModelInfo);
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to get models: ${error}`);
    }
  }

  /**
   * Get model information
   */
  async getModelInfo(modelName: string): Promise<ModelInfo> {
    try {
      const response = await this.client.get(`/models/${modelName}`);
      return this.transformModelInfo(response.data);
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to get model info: ${error}`);
    }
  }

  /**
   * Get API health status
   */
  async getHealth(): Promise<any> {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Health check failed: ${error}`);
    }
  }

  /**
   * Get API metrics
   */
  async getMetrics(): Promise<any> {
    try {
      const response = await this.client.get('/metrics');
      return response.data;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to get metrics: ${error}`);
    }
  }

  private transformResponse(data: any): PredictionResponse {
    return {
      prediction: {
        primaryNotes: data.prediction.primary_notes,
        descriptors: data.prediction.descriptors,
        intensity: data.prediction.intensity,
        confidence: data.prediction.confidence,
        chemicalFamily: data.prediction.chemical_family,
        categories: data.prediction.categories || []
      },
      molecularFeatures: data.molecular_features ? {
        molecularWeight: data.molecular_features.molecular_weight,
        logp: data.molecular_features.logp,
        tpsa: data.molecular_features.tpsa,
        numAtoms: data.molecular_features.num_atoms,
        numBonds: data.molecular_features.num_bonds,
        numRings: data.molecular_features.num_rings,
        aromaticRings: data.molecular_features.aromatic_rings,
        rotatableBonds: data.molecular_features.rotatable_bonds
      } : undefined,
      attentionWeights: data.attention_weights ? {
        layerWeights: data.attention_weights.layer_weights,
        headWeights: data.attention_weights.head_weights,
        tokenWeights: data.attention_weights.token_weights
      } : undefined,
      metadata: data.metadata,
      processingTimeMs: data.processing_time_ms,
      modelVersion: data.model_version,
      cached: data.cached
    };
  }

  private transformBatchResponse(data: any): BatchPredictionResponse {
    return {
      results: data.results.map((result: any) => ({
        molecule: result.molecule,
        prediction: result.prediction ? {
          primaryNotes: result.prediction.primary_notes,
          descriptors: result.prediction.descriptors,
          intensity: result.prediction.intensity,
          confidence: result.prediction.confidence,
          chemicalFamily: result.prediction.chemical_family,
          categories: result.prediction.categories || []
        } : undefined,
        error: result.error,
        success: result.success,
        processingTimeMs: result.processing_time_ms
      })),
      summary: data.summary,
      totalProcessingTimeMs: data.total_processing_time_ms,
      batchId: data.batch_id
    };
  }

  private transformModelInfo(data: any): ModelInfo {
    return {
      name: data.name,
      version: data.version,
      status: data.status,
      parameters: data.parameters,
      architecture: data.architecture,
      capabilities: data.capabilities,
      trainingData: data.training_data,
      performanceMetrics: data.performance_metrics,
      lastUpdated: new Date(data.last_updated)
    };
  }

  /**
   * Close all connections and cleanup
   */
  async close(): Promise<void> {
    // Close all active streams
    for (const [sessionId, ws] of this.activeStreams) {
      ws.close();
    }
    this.activeStreams.clear();

    this.removeAllListeners();
  }
}

/**
 * Factory function to create client
 */
export function createClient(config: ClientConfig): OlfactoryClient {
  return new OlfactoryClient(config);
}

// Re-export types
export * from './types';

// Default export
export default OlfactoryClient;