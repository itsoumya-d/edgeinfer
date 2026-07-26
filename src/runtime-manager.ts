import * as ort from 'onnxruntime-web';

export class RuntimeManager {
  static async getBestExecutionProvider(): Promise<string> {
    if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
      return 'webgpu';
    }
    return 'wasm';
  }

  static async createSession(
    modelBuffer: ArrayBuffer,
    providers?: string[]
  ): Promise<ort.InferenceSession> {
    const eps = providers || [await this.getBestExecutionProvider()];
    return await ort.InferenceSession.create(modelBuffer, { executionProviders: eps });
  }
}
