// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com | +91 7031648617

import * as ort from 'onnxruntime-web';
import { ModelCache } from './model-cache';
import { RuntimeManager } from './runtime-manager';
import { ModelOptions, ClassificationResult, Detection } from './types';
import { Tokenizer } from './tokenizer';
import { ImageProcessor } from './image-processor';

export class EdgeInfer {
  private session: ort.InferenceSession;
  private _inputNames: readonly string[];
  private _outputNames: readonly string[];
  private _modelSize: number;

  private constructor(session: ort.InferenceSession, modelSize: number) {
    this.session = session;
    this._inputNames = session.inputNames;
    this._outputNames = session.outputNames;
    this._modelSize = modelSize;
  }

  static async load(modelUrl: string, options?: ModelOptions): Promise<EdgeInfer> {
    const cache = new ModelCache(options?.cacheName);
    const buffer = await cache.getModel(modelUrl, options?.forceDownload);
    return EdgeInfer.fromBuffer(buffer, options);
  }

  static async fromBuffer(buffer: ArrayBuffer, options?: ModelOptions): Promise<EdgeInfer> {
    const session = await RuntimeManager.createSession(buffer, options?.executionProviders);
    return new EdgeInfer(session, buffer.byteLength);
  }

  async predict(inputs: Record<string, Float32Array | Int32Array>): Promise<Record<string, Float32Array>> {
    const tensorInputs: Record<string, ort.Tensor> = {};
    
    for (const [key, data] of Object.entries(inputs)) {
      const type = data instanceof Float32Array ? 'float32' : 'int32';
      tensorInputs[key] = new ort.Tensor(type, data, [1, data.length]);
    }

    const results = await this.session.run(tensorInputs);
    const outputMap: Record<string, Float32Array> = {};

    for (const [key, tensor] of Object.entries(results)) {
      outputMap[key] = tensor.data as Float32Array;
    }

    return outputMap;
  }

  async classify(text: string): Promise<ClassificationResult[]> {
    // Basic implementation assuming a text input model
    const tokenizer = new Tokenizer({});
    const inputIds = tokenizer.encode(text);
    const result = await this.predict({ 'input_ids': inputIds });
    const output = result[this.outputNames[0]];
    
    // Convert output tensor to probabilities/scores and labels
    return Array.from(output || []).map((score, idx) => ({
      label: `Class_${idx}`,
      score
    }));
  }

  async embed(text: string): Promise<Float32Array> {
    const tokenizer = new Tokenizer({});
    const inputIds = tokenizer.encode(text);
    const result = await this.predict({ 'input_ids': inputIds });
    return result[this.outputNames[0]] || new Float32Array();
  }

  async sentiment(text: string): Promise<{ label: string; score: number }> {
    const results = await this.classify(text);
    const best = results.reduce((prev, current) => (prev.score > current.score) ? prev : current, { label: 'unknown', score: 0 });
    return {
      label: best.label === 'Class_1' ? 'positive' : 'negative',
      score: best.score
    };
  }

  async classifyImage(imageData: ImageData): Promise<ClassificationResult[]> {
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData);
    const result = await this.predict({ 'input': tensorData });
    const output = result[this.outputNames[0]];
    
    return Array.from(output || []).map((score, idx) => ({
      label: `Class_${idx}`,
      score
    })).sort((a, b) => b.score - a.score).slice(0, 5);
  }

  async detectObjects(imageData: ImageData): Promise<Detection[]> {
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData);
    const result = await this.predict({ 'input': tensorData });
    // Assuming output format is [x, y, w, h, score, class_id] for each detection
    const output = result[this.outputNames[0]];
    const detections: Detection[] = [];
    
    if (output) {
      for (let i = 0; i < output.length; i += 6) {
        if (output[i + 4] > 0.5) {
          detections.push({
            bbox: [output[i], output[i+1], output[i+2], output[i+3]],
            score: output[i+4],
            label: `Object_${Math.round(output[i+5])}`
          });
        }
      }
    }
    return detections;
  }

  get inputNames(): readonly string[] { return this._inputNames; }
  get outputNames(): readonly string[] { return this._outputNames; }
  get modelSize(): number { return this._modelSize; }

  dispose(): void {
    // Release session resources (if any direct method exists or garbage collect)
  }
}
