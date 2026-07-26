import * as ort from 'onnxruntime-web';
import { ModelCache } from './model-cache';
import { RuntimeManager } from './runtime-manager';
import { ModelOptions, ClassificationResult, Detection } from './types';

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
    throw new Error('Not implemented');
  }

  async embed(text: string): Promise<Float32Array> {
    throw new Error('Not implemented');
  }

  async sentiment(text: string): Promise<{ label: string; score: number }> {
    throw new Error('Not implemented');
  }

  async classifyImage(imageData: ImageData): Promise<ClassificationResult[]> {
    throw new Error('Not implemented');
  }

  async detectObjects(imageData: ImageData): Promise<Detection[]> {
    throw new Error('Not implemented');
  }

  get inputNames(): readonly string[] { return this._inputNames; }
  get outputNames(): readonly string[] { return this._outputNames; }
  get modelSize(): number { return this._modelSize; }

  dispose(): void {
    // Release session resources (if any direct method exists or garbage collect)
  }
}
