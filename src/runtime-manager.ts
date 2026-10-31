// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

import * as ort from 'onnxruntime-web';

export interface VRAMProbeResult {
  available: boolean;
  /** Largest single storage buffer the device actually allocated. */
  maxAllocationBytes: number;
  testedAllocations: number[];
  failedAllocationBytes: number | null;
  adapterLimits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
  };
  deviceCreated: boolean;
  probe: 'empirical' | 'limits' | 'unavailable';
}

export interface VRAMProbeOptions {
  startBytes?: number;
  maxBytes?: number;
  stepBytes?: number;
}

export interface GPUCapabilities {
  provider: string;
  hasWebGPU: boolean;
  hasWebNN: boolean;
  /**
   * NOT an amount of video memory. This is the largest single GPU storage
   * buffer allocated by the empirical probe, bounded by the adapter's
   * `maxBufferSize`. Browser baselines are commonly 256 MiB regardless of the
   * physical card, so treat this as a maximum-tensor-size budget, not as VRAM.
   */
  estimatedVRAM: number;
  /** Raw adapter limits, kept separate from the empirical probe. */
  adapterLimits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
  } | null;
  /** Evidence for how `estimatedVRAM` was obtained. */
  vramProbe: VRAMProbeResult | null;
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
      adapterLimits: null,
      vramProbe: null,
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

          const maxBufferSize = adapter.limits?.maxBufferSize || 0;
          const maxStorageSize = adapter.limits?.maxStorageBufferBindingSize || 0;
          capabilities.adapterLimits = {
            maxBufferSize,
            maxStorageBufferBindingSize: maxStorageSize,
          };

          // Probe what the device actually allocates instead of treating the
          // adapter's baseline limits as physical VRAM.
          const probe = await RuntimeManager.probeVRAM(adapter);
          capabilities.vramProbe = probe;
          capabilities.estimatedVRAM = probe.available
            ? probe.maxAllocationBytes
            : Math.max(maxBufferSize, maxStorageSize);

          const vramGB = capabilities.estimatedVRAM / (1024 * 1024 * 1024);
          if (vramGB >= 4) {
            capabilities.recommendedQuantization = 'fp16';
          } else if (vramGB >= 1) {
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
   * Empirically probe the largest single storage buffer the adapter's device
   * will allocate. Allocates and destroys one buffer at a time; never relies on
   * `maxBufferSize` as if it were physical VRAM.
   */
  static async probeVRAM(
    adapter: any,
    options: VRAMProbeOptions = {}
  ): Promise<VRAMProbeResult> {
    const limits = {
      maxBufferSize: Number(adapter?.limits?.maxBufferSize) || 0,
      maxStorageBufferBindingSize:
        Number(adapter?.limits?.maxStorageBufferBindingSize) || 0,
    };
    const unavailable: VRAMProbeResult = {
      available: false,
      maxAllocationBytes: 0,
      testedAllocations: [],
      failedAllocationBytes: null,
      adapterLimits: limits,
      deviceCreated: false,
      probe: 'unavailable',
    };

    if (!adapter || typeof adapter.requestDevice !== 'function') return unavailable;

    let device: any;
    try {
      device = await adapter.requestDevice();
    } catch {
      return unavailable;
    }
    if (!device || typeof device.createBuffer !== 'function') {
      return { ...unavailable, deviceCreated: Boolean(device) };
    }

    const MIB = 1024 * 1024;
    const start = Math.max(1, options.startBytes ?? 64 * MIB);
    const configuredMax = Math.max(start, options.maxBytes ?? 4 * 1024 * MIB);
    const max = limits.maxBufferSize > 0
      ? Math.min(configuredMax, limits.maxBufferSize)
      : configuredMax;
    const step = Math.max(1, options.stepBytes ?? 32 * MIB);
    const tested: number[] = [];
    const STORAGE = 0x80;
    const COPY_DST = 0x8;

    const attempt = (size: number): boolean => {
      tested.push(size);
      try {
        const buffer = device.createBuffer({ size, usage: STORAGE | COPY_DST });
        buffer?.destroy?.();
        return true;
      } catch {
        return false;
      }
    };

    let low = 0;
    let high = 0;
    let size = Math.min(start, max);
    while (size >= 1) {
      if (attempt(size)) {
        low = size;
        break;
      }
      high = size;
      if (size === 1) break;
      size = Math.max(1, Math.floor(size / 2));
    }

    if (low === 0) {
      device.destroy?.();
      return {
        ...unavailable,
        testedAllocations: tested,
        failedAllocationBytes: high || null,
        deviceCreated: true,
        probe: 'limits',
      };
    }

    let next = Math.min(max, low * 2);
    while (next > low && tested.length < 16) {
      if (attempt(next)) {
        low = next;
        next = Math.min(max, low * 2);
        if (low === max) break;
      } else {
        high = next;
        break;
      }
    }
    if (high === 0) high = Math.min(max, low * 2);

    while (high - low > step && tested.length < 24) {
      const mid = Math.floor((low + high) / 2);
      if (mid <= low || mid >= high) break;
      if (attempt(mid)) low = mid;
      else high = mid;
    }

    device.destroy?.();
    return {
      available: true,
      maxAllocationBytes: low,
      testedAllocations: tested,
      failedAllocationBytes: high > low ? high : null,
      adapterLimits: limits,
      deviceCreated: true,
      probe: 'empirical',
    };
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
