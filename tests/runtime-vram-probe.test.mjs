// Empirical WebGPU buffer-allocation probe. Uses a mocked adapter so the binary
// search is deterministic and requires no GPU.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const { RuntimeManager } = await import(join(__dirname, '../dist/index.mjs'));

const MIB = 1024 * 1024;

function mockAdapter(maxAllocationBytes, { requestDeviceFails = false } = {}) {
  const destroyed = [];
  const adapter = {
    limits: {
      maxBufferSize: 256 * MIB,
      maxStorageBufferBindingSize: 128 * MIB,
    },
    async requestDevice() {
      if (requestDeviceFails) throw new Error('device creation failed');
      return {
        createBuffer({ size }) {
          if (size > maxAllocationBytes) throw new Error('allocation failed');
          return { destroy() { destroyed.push(size); } };
        },
        destroy() { destroyed.push('device'); }
      };
    }
  };
  return { adapter, destroyed };
}

describe('RuntimeManager.probeVRAM', () => {
  test('binary-searches the largest allocatable storage buffer', async () => {
    const { adapter, destroyed } = mockAdapter(100 * MIB);
    const result = await RuntimeManager.probeVRAM(adapter, {
      startBytes: 8 * MIB,
      maxBytes: 128 * MIB,
      stepBytes: 4 * MIB
    });

    assert.equal(result.probe, 'empirical');
    assert.equal(result.available, true);
    assert.equal(result.maxAllocationBytes, 100 * MIB);
    assert.ok(result.testedAllocations.includes(64 * MIB));
    assert.ok(result.testedAllocations.includes(128 * MIB));
    assert.ok(result.failedAllocationBytes >= 100 * MIB);
    assert.equal(result.adapterLimits.maxBufferSize, 256 * MIB);
    assert.equal(result.deviceCreated, true);
    assert.ok(destroyed.includes('device'));
  });

  test('falls back to limits when every allocation fails', async () => {
    const { adapter } = mockAdapter(0);
    const result = await RuntimeManager.probeVRAM(adapter, {
      startBytes: 8 * MIB,
      maxBytes: 32 * MIB,
      stepBytes: 4 * MIB
    });
    assert.equal(result.probe, 'limits');
    assert.equal(result.available, false);
    assert.equal(result.maxAllocationBytes, 0);
    assert.ok(result.testedAllocations.length > 0);
  });

  test('reports unavailable when device creation fails or no adapter exists', async () => {
    const { adapter } = mockAdapter(10 * MIB, { requestDeviceFails: true });
    const failed = await RuntimeManager.probeVRAM(adapter);
    assert.equal(failed.probe, 'unavailable');
    assert.equal(failed.available, false);

    const missing = await RuntimeManager.probeVRAM(null);
    assert.equal(missing.probe, 'unavailable');
  });

  test('detectCapabilities uses the probe instead of adapter limits as VRAM', async () => {
    const { adapter } = mockAdapter(100 * MIB);
    const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { gpu: { requestAdapter: async () => adapter } }
    });
    RuntimeManager.resetCache();
    try {
      const caps = await RuntimeManager.detectCapabilities();
      assert.equal(caps.hasWebGPU, true);
      assert.equal(caps.provider, 'webgpu');
      assert.equal(caps.estimatedVRAM, caps.vramProbe.maxAllocationBytes);
      assert.ok(caps.estimatedVRAM >= 96 * MIB && caps.estimatedVRAM <= 100 * MIB);
      assert.equal(caps.vramProbe.probe, 'empirical');
      assert.equal(caps.adapterLimits.maxBufferSize, 256 * MIB);
      // 100 MiB is below the 1 GiB int8 threshold.
      assert.equal(caps.recommendedQuantization, 'int4');
    } finally {
      RuntimeManager.resetCache();
      if (original) Object.defineProperty(globalThis, 'navigator', original);
      else delete globalThis.navigator;
    }
  });
});
