// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com

import * as ort from 'onnxruntime-web';

export interface GPUCapabilities {
  provider: string;
  hasWebGPU: boolean;
  hasWebNN: boolean;
  /**
   * NOT an amount of video memory. This is the larger of the adapter's
   * `maxBufferSize` and `maxStorageBufferBindingSize` limits, i.e. the largest
   * single GPU buffer/binding WebGPU will allow. The WebGPU spec's guaranteed
   * baseline values are 256 MiB and 128 MiB respectively, and browsers commonly
   * report values close to the baseline regardless of how much VRAM the card
   * physically has. Treat this as a maximum-tensor-size budget, not as VRAM.
   */
  estimatedVRAM: number;
  recommendedQuantization: 'fp32' | 'fp16' | 'int8' | 'int4';
}

/** Max time to wait for navigator.gpu.requestAdapter() before giving up. */
const ADAPTER_TIMEOUT_MS = 5000;

/**
 * RuntimeManager handles execution provider detection and ONNX session creation.
 * Implements a smart fallback chain: WebGPU → WebNN → WASM
 * with automatic quantization level recommendation based on available GPU memory.
 */
export class RuntimeManager {
  private static cachedCapabilities: GPUCapabilities | null = null;

  /**
   * Detect GPU capabilities and recommend the best execution configuration.
   */
  static async detectCapabilities(): Promise<GPUCapabilities> {
    if (this.cachedCapabilities) return this.cachedCapabilities;

    const capabilities: GPUCapabilities = {
      provider: 'wasm',
      hasWebGPU: false,
      hasWebNN: false,
      estimatedVRAM: 0,
      recommendedQuantization: 'int8',
    };

    // Check WebGPU availability.
    // NOTE: `'gpu' in navigator` is NOT sufficient to conclude WebGPU works.
    // A browser can expose navigator.gpu and still return null from
    // requestAdapter() (no GPU, driver blocklisted, headless). We therefore
    // always request an adapter, and we bound the wait: requestAdapter() is
    // specified to settle, but a hung driver must not stall the whole SDK.
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const gpu = (navigator as any).gpu;
        const adapter = await RuntimeManager.withTimeout(
          Promise.resolve(gpu.requestAdapter()),
          ADAPTER_TIMEOUT_MS
        );
        if (adapter) {
          capabilities.hasWebGPU = true;
          capabilities.provider = 'webgpu';

          // Estimate VRAM from adapter limits
          const maxBufferSize = adapter.limits?.maxBufferSize || 0;
          const maxStorageSize = adapter.limits?.maxStorageBufferBindingSize || 0;
          capabilities.estimatedVRAM = Math.max(maxBufferSize, maxStorageSize);

          // Recommend quantization based on estimated VRAM
          const vramGB = capabilities.estimatedVRAM / (1024 * 1024 * 1024);
          if (vramGB >= 8) {
            capabilities.recommendedQuantization = 'fp16';
          } else if (vramGB >= 4) {
            capabilities.recommendedQuantization = 'int8';
          } else {
            capabilities.recommendedQuantization = 'int4';
          }
        }
      } catch {
        // WebGPU not available or adapter request failed
      }
    }

    // Check WebNN availability
    if (typeof navigator !== 'undefined' && 'ml' in navigator) {
      try {
        const ml = (navigator as any).ml;
        if (ml) {
          capabilities.hasWebNN = true;
          if (!capabilities.hasWebGPU) {
            capabilities.provider = 'webnn';
          }
        }
      } catch {
        // WebNN not available
      }
    }

    this.cachedCapabilities = capabilities;
    return capabilities;
  }

  /**
   * Get the best available execution provider with smart fallback.
   * Priority: WebGPU → WebNN → WASM
   */
  static async getBestExecutionProvider(): Promise<string> {
    const caps = await this.detectCapabilities();
    return caps.provider;
  }

  /**
   * Create an ONNX inference session with automatic provider selection.
   * Falls back gracefully if the preferred provider fails.
   *
   * Returns BOTH the session and the provider that actually served it. The
   * detected capability (`detectCapabilities().provider`) is only a preference:
   * a provider can be detected as available and still fail to create a session
   * (for example onnxruntime-web only registers the WebGPU execution provider
   * in some builds/versions). Callers must report the returned `provider`, not
   * the detected one, or `EdgeInfer.executionProvider` will lie.
   */
  static async createSession(
    modelBuffer: ArrayBuffer,
    providers?: string[]
  ): Promise<{ session: ort.InferenceSession; provider: string }> {
    const requestedProviders = providers || [await this.getBestExecutionProvider()];

    // Try each provider in order, falling back on failure
    for (const provider of requestedProviders) {
      try {
        const session = await ort.InferenceSession.create(
          modelBuffer,
          {
            executionProviders: [provider],
            graphOptimizationLevel: 'all',
          }
        );
        return { session, provider };
      } catch (err) {
        console.warn(`EdgeInfer: Failed to create session with provider "${provider}", trying next...`, err);
      }
    }

    // Final fallback: WASM (should always work)
    console.warn('EdgeInfer: All preferred providers failed, falling back to WASM');
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return { session, provider: 'wasm' };
  }

  /**
   * Resolve `promise`, or resolve to `null` if it has not settled within `ms`.
   * Used to stop a hung GPU driver from blocking model loading forever.
   */
  private static withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | null> {
    let timer: any;
    return Promise.race([
      promise.then((v) => { clearTimeout(timer); return v; }),
      new Promise<null>((resolve) => {
        timer = setTimeout(() => {
          console.warn(`EdgeInfer: navigator.gpu.requestAdapter() did not settle within ${ms}ms; falling back to WASM.`);
          resolve(null);
        }, ms);
        if (typeof timer?.unref === 'function') timer.unref();
      }),
    ]);
  }

  /**
   * Estimate if a model of given size (bytes) can fit in available VRAM.
   */
  static async canFitModel(modelSizeBytes: number): Promise<boolean> {
    const caps = await this.detectCapabilities();
    if (!caps.hasWebGPU) return true; // WASM uses system RAM, not VRAM
    // Model typically needs 1.5x its size in VRAM (weights + activations)
    return caps.estimatedVRAM >= modelSizeBytes * 1.5;
  }

  /**
   * Reset cached capabilities (useful for testing or re-detection).
   */
  static resetCache(): void {
    this.cachedCapabilities = null;
  }
}
