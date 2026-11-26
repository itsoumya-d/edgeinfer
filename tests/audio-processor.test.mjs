// SPDX-License-Identifier: MIT
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const {
  AudioProcessor,
  StreamingAudioBuffer,
} = await import(pathToFileURL(join(__dirname, '../dist/index.mjs')).href);

describe('AudioProcessor & StreamingAudioBuffer', () => {
  test('exports AudioProcessor and StreamingAudioBuffer classes', () => {
    assert.strictEqual(typeof AudioProcessor, 'function');
    assert.strictEqual(typeof StreamingAudioBuffer, 'function');
  });

  test('hzToMel and melToHz invertible conversion', () => {
    const f0 = 0;
    const m0 = AudioProcessor.hzToMel(f0);
    assert.strictEqual(m0, 0);
    assert.strictEqual(AudioProcessor.melToHz(m0), 0);

    const testFreqs = [100, 440, 1000, 4000, 8000];
    for (const freq of testFreqs) {
      const mel = AudioProcessor.hzToMel(freq);
      const back = AudioProcessor.melToHz(mel);
      assert.ok(Math.abs(back - freq) < 1e-4, `Expected ${freq}, got ${back}`);
    }
  });

  test('applyPreEmphasis filters signal correctly', () => {
    const single = new Float32Array([1.5]);
    const filteredSingle = AudioProcessor.applyPreEmphasis(single, 0.97);
    assert.deepStrictEqual(Array.from(filteredSingle), [1.5]);

    const signal = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const noEmphasis = AudioProcessor.applyPreEmphasis(signal, 0.0);
    assert.deepStrictEqual(Array.from(noEmphasis), [1.0, 2.0, 3.0, 4.0]);

    const filtered = AudioProcessor.applyPreEmphasis(signal, 0.5);
    // [1.0, 2.0 - 0.5*1.0, 3.0 - 0.5*2.0, 4.0 - 0.5*3.0] = [1.0, 1.5, 2.0, 2.5]
    assert.strictEqual(filtered.length, 4);
    assert.strictEqual(filtered[0], 1.0);
    assert.strictEqual(filtered[1], 1.5);
    assert.strictEqual(filtered[2], 2.0);
    assert.strictEqual(filtered[3], 2.5);
  });

  test('getWindow handles rectangular, hann, and hamming windows', () => {
    const rect = AudioProcessor.getWindow(10, 'rectangular');
    assert.strictEqual(rect.length, 10);
    assert.ok(Array.from(rect).every((v) => v === 1.0));

    const small = AudioProcessor.getWindow(1, 'hann');
    assert.strictEqual(small[0], 1.0);

    const hann = AudioProcessor.getWindow(32, 'hann');
    assert.strictEqual(hann[0], 0.0);
    assert.ok(Math.abs(hann[hann.length - 1]) < 1e-6);
    assert.ok(hann[16] > 0.99);

    const hamming = AudioProcessor.getWindow(32, 'hamming');
    assert.ok(Math.abs(hamming[0] - 0.08) < 1e-3);
    assert.ok(Math.abs(hamming[hamming.length - 1] - 0.08) < 1e-3);
    assert.ok(hamming[16] > 0.99);
  });

  test('frameAudio handles valid and invalid framing configurations', () => {
    assert.throws(() => {
      AudioProcessor.frameAudio(new Float32Array(100), { frameSize: -1 });
    }, /Invalid frameSize/);

    assert.throws(() => {
      AudioProcessor.frameAudio(new Float32Array(100), { hopSize: 0 });
    }, /Invalid frameSize/);

    const samples = new Float32Array(400);
    for (let i = 0; i < samples.length; i++) samples[i] = 1.0;

    const frames = AudioProcessor.frameAudio(samples, {
      frameSize: 100,
      hopSize: 50,
      window: 'rectangular',
      preEmphasis: 0,
    });
    // start indices: 0, 50, 100, 150, 200, 250, 300 (total 7 frames)
    assert.strictEqual(frames.length, 7);
    assert.strictEqual(frames[0].length, 100);
  });

  test('computePowerSpectrum isolates frequency component', () => {
    const fftSize = 64;
    const numBins = 33;
    const frame = new Float32Array(fftSize);

    // DC signal
    frame.fill(1.0);
    const dcPower = AudioProcessor.computePowerSpectrum(frame, fftSize);
    assert.strictEqual(dcPower.length, numBins);
    assert.ok(dcPower[0] > 0);
    // Non-DC bins should be near 0
    for (let k = 1; k < numBins; k++) {
      assert.ok(dcPower[k] < 1e-6);
    }

    // Pure cosine at bin index 4
    const kTarget = 4;
    for (let n = 0; n < fftSize; n++) {
      frame[n] = Math.cos((2 * Math.PI * kTarget * n) / fftSize);
    }
    const tonePower = AudioProcessor.computePowerSpectrum(frame, fftSize);
    assert.ok(tonePower[kTarget] > 100);
    for (let k = 0; k < numBins; k++) {
      if (k !== kTarget) {
        assert.ok(tonePower[k] < 1e-6, `Bin ${k} power was ${tonePower[k]}`);
      }
    }
  });

  test('createMelFilterbank produces correct number of triangular filters', () => {
    const numMelBins = 40;
    const fftSize = 512;
    const filterbank = AudioProcessor.createMelFilterbank(numMelBins, 16000, fftSize, 0, 8000);

    assert.strictEqual(filterbank.length, numMelBins);
    const numFreqBins = Math.floor(fftSize / 2) + 1; // 257
    for (const filter of filterbank) {
      assert.strictEqual(filter.length, numFreqBins);
      // Peak weight is <= 1.0 and >= 0.0
      const maxVal = Math.max(...filter);
      assert.ok(maxVal <= 1.0 && maxVal > 0.0);
    }
  });

  test('extractMelSpectrogram extracts tensor with expected layout and dimensions', () => {
    // Empty samples test
    const empty = AudioProcessor.extractMelSpectrogram(new Float32Array(0), {
      numMelBins: 40,
      layout: 'NTC',
    });
    assert.strictEqual(empty.numFrames, 0);
    assert.deepStrictEqual(empty.shape, [1, 0, 40]);
    assert.strictEqual(empty.data.length, 0);

    // Empty samples in NCT layout
    const emptyNCT = AudioProcessor.extractMelSpectrogram(new Float32Array(0), {
      numMelBins: 40,
      layout: 'NCT',
    });
    assert.deepStrictEqual(emptyNCT.shape, [1, 40, 0]);

    // Synthetic audio samples (16000 Hz, 0.1s = 1600 samples)
    const sampleRate = 16000;
    const samples = new Float32Array(1600);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const ntc = AudioProcessor.extractMelSpectrogram(samples, {
      sampleRate,
      frameSize: 400,
      hopSize: 160,
      numMelBins: 40,
      fftSize: 512,
      layout: 'NTC',
      logMel: true,
    });
    // frames: start at 0, 160, 320, 480, 640, 800, 960, 1120. (1120 + 400 = 1520 <= 1600). Total 8 frames.
    assert.strictEqual(ntc.numFrames, 8);
    assert.strictEqual(ntc.numBins, 40);
    assert.deepStrictEqual(ntc.shape, [1, 8, 40]);
    assert.strictEqual(ntc.data.length, 8 * 40);
    assert.ok(!Number.isNaN(ntc.data[0]));

    // NCT layout test
    const nct = AudioProcessor.extractMelSpectrogram(samples, {
      sampleRate,
      frameSize: 400,
      hopSize: 160,
      numMelBins: 40,
      fftSize: 512,
      layout: 'NCT',
      logMel: false,
    });
    assert.deepStrictEqual(nct.shape, [1, 40, 8]);
    assert.strictEqual(nct.data.length, 40 * 8);
  });

  test('StreamingAudioBuffer maintains ring buffer and yields complete frames', () => {
    assert.throws(() => new StreamingAudioBuffer(0, 160));
    assert.throws(() => new StreamingAudioBuffer(400, -5));

    const stream = new StreamingAudioBuffer(10, 5);
    assert.strictEqual(stream.length, 0);

    // Push empty chunk
    assert.deepStrictEqual(stream.push(new Float32Array(0)), []);

    // Push 8 samples (less than frameSize 10)
    const chunk1 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const frames1 = stream.push(chunk1);
    assert.strictEqual(frames1.length, 0);
    assert.strictEqual(stream.length, 8);

    // Push another 8 samples (total 16 samples)
    // Frame 1: 0..9 (indices 0..9)
    // Frame 2: 5..14 (indices 5..14)
    // Leftover: 10..15 (6 samples)
    const chunk2 = new Float32Array([9, 10, 11, 12, 13, 14, 15, 16]);
    const frames2 = stream.push(chunk2);
    assert.strictEqual(frames2.length, 2);
    assert.deepStrictEqual(Array.from(frames2[0]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert.deepStrictEqual(Array.from(frames2[1]), [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert.strictEqual(stream.length, 6);

    // Flush remaining
    const remaining = stream.flush();
    assert.ok(remaining !== null);
    assert.deepStrictEqual(Array.from(remaining), [11, 12, 13, 14, 15, 16]);
    assert.strictEqual(stream.length, 0);
    assert.strictEqual(stream.flush(), null);

    // Push and reset
    stream.push(new Float32Array([1, 2, 3]));
    assert.strictEqual(stream.length, 3);
    stream.reset();
    assert.strictEqual(stream.length, 0);
  });
});
