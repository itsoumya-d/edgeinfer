// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

export interface AudioFramingOptions {
  frameSize?: number;
  hopSize?: number;
  window?: 'hann' | 'hamming' | 'rectangular';
  preEmphasis?: number;
}

export interface MelSpectrogramOptions extends AudioFramingOptions {
  sampleRate?: number;
  fftSize?: number;
  numMelBins?: number;
  minFreq?: number;
  maxFreq?: number;
  logMel?: boolean;
  epsilon?: number;
  layout?: 'NTC' | 'NCT';
}

export interface AudioTensorOutput {
  data: Float32Array;
  shape: [number, number, number];
  numFrames: number;
  numBins: number;
}

export class StreamingAudioBuffer {
  private buffer: Float32Array;
  readonly frameSize: number;
  readonly hopSize: number;

  constructor(frameSize: number = 400, hopSize: number = 160) {
    if (frameSize <= 0 || hopSize <= 0) {
      throw new Error(`Invalid frameSize (${frameSize}) or hopSize (${hopSize})`);
    }
    this.frameSize = frameSize;
    this.hopSize = hopSize;
    this.buffer = new Float32Array(0);
  }

  get length(): number {
    return this.buffer.length;
  }

  reset(): void {
    this.buffer = new Float32Array(0);
  }

  push(chunk: Float32Array): Float32Array[] {
    if (chunk.length === 0) return [];

    const merged = new Float32Array(this.buffer.length + chunk.length);
    merged.set(this.buffer, 0);
    merged.set(chunk, this.buffer.length);

    const frames: Float32Array[] = [];
    let offset = 0;
    while (offset + this.frameSize <= merged.length) {
      frames.push(merged.slice(offset, offset + this.frameSize));
      offset += this.hopSize;
    }

    if (offset < merged.length) {
      this.buffer = merged.slice(offset);
    } else {
      this.buffer = new Float32Array(0);
    }

    return frames;
  }

  flush(): Float32Array | null {
    if (this.buffer.length === 0) return null;
    const remaining = this.buffer;
    this.reset();
    return remaining;
  }
}

export class AudioProcessor {
  static hzToMel(hz: number): number {
    return 2595 * Math.log10(1 + hz / 700);
  }

  static melToHz(mel: number): number {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }

  static applyPreEmphasis(samples: Float32Array, alpha: number = 0.97): Float32Array {
    if (alpha === 0 || samples.length <= 1) {
      return new Float32Array(samples);
    }
    const out = new Float32Array(samples.length);
    out[0] = samples[0];
    for (let i = 1; i < samples.length; i++) {
      out[i] = samples[i] - alpha * samples[i - 1];
    }
    return out;
  }

  static getWindow(size: number, type: 'hann' | 'hamming' | 'rectangular' = 'hann'): Float32Array {
    const win = new Float32Array(size);
    if (type === 'rectangular' || size <= 1) {
      win.fill(1.0);
      return win;
    }
    const denom = size - 1;
    for (let i = 0; i < size; i++) {
      if (type === 'hann') {
        win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / denom));
      } else {
        win[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / denom);
      }
    }
    return win;
  }

  static frameAudio(
    samples: Float32Array,
    options: AudioFramingOptions = {}
  ): Float32Array[] {
    const {
      frameSize = 400,
      hopSize = 160,
      window = 'hann',
      preEmphasis = 0.97,
    } = options;

    if (frameSize <= 0 || hopSize <= 0) {
      throw new Error(`Invalid frameSize (${frameSize}) or hopSize (${hopSize})`);
    }

    const filtered = AudioProcessor.applyPreEmphasis(samples, preEmphasis);
    const win = AudioProcessor.getWindow(frameSize, window);

    const frames: Float32Array[] = [];
    for (let start = 0; start + frameSize <= filtered.length; start += hopSize) {
      const frame = new Float32Array(frameSize);
      for (let i = 0; i < frameSize; i++) {
        frame[i] = filtered[start + i] * win[i];
      }
      frames.push(frame);
    }
    return frames;
  }

  static computePowerSpectrum(frame: Float32Array, fftSize: number = 512): Float32Array {
    const numBins = Math.floor(fftSize / 2) + 1;
    const power = new Float32Array(numBins);
    const frameLen = Math.min(frame.length, fftSize);

    for (let k = 0; k < numBins; k++) {
      let real = 0;
      let imag = 0;
      const angleStep = (2 * Math.PI * k) / fftSize;
      for (let n = 0; n < frameLen; n++) {
        const angle = angleStep * n;
        real += frame[n] * Math.cos(angle);
        imag -= frame[n] * Math.sin(angle);
      }
      power[k] = real * real + imag * imag;
    }
    return power;
  }

  static createMelFilterbank(
    numMelBins: number = 80,
    sampleRate: number = 16000,
    fftSize: number = 512,
    minFreq: number = 0,
    maxFreq?: number
  ): Float32Array[] {
    const topFreq = maxFreq ?? sampleRate / 2;
    const numFreqBins = Math.floor(fftSize / 2) + 1;

    const minMel = AudioProcessor.hzToMel(minFreq);
    const maxMel = AudioProcessor.hzToMel(topFreq);
    const melPoints = new Float32Array(numMelBins + 2);
    const hzPoints = new Float32Array(numMelBins + 2);
    const binPoints = new Int32Array(numMelBins + 2);

    for (let i = 0; i < numMelBins + 2; i++) {
      melPoints[i] = minMel + (i * (maxMel - minMel)) / (numMelBins + 1);
      hzPoints[i] = AudioProcessor.melToHz(melPoints[i]);
      binPoints[i] = Math.min(
        numFreqBins - 1,
        Math.floor(((fftSize + 1) * hzPoints[i]) / sampleRate)
      );
    }

    const filterbank: Float32Array[] = [];
    for (let m = 1; m <= numMelBins; m++) {
      const filter = new Float32Array(numFreqBins);
      const left = binPoints[m - 1];
      const center = binPoints[m];
      const right = binPoints[m + 1];

      for (let k = left; k < center; k++) {
        if (center > left) {
          filter[k] = (k - left) / (center - left);
        }
      }
      for (let k = center; k <= right; k++) {
        if (right > center) {
          filter[k] = (right - k) / (right - center);
        }
      }
      filterbank.push(filter);
    }
    return filterbank;
  }

  static extractMelSpectrogram(
    samples: Float32Array,
    options: MelSpectrogramOptions = {}
  ): AudioTensorOutput {
    const {
      sampleRate = 16000,
      fftSize = 512,
      numMelBins = 80,
      minFreq = 0,
      maxFreq = 8000,
      logMel = true,
      epsilon = 1e-6,
      layout = 'NTC',
    } = options;

    const frames = AudioProcessor.frameAudio(samples, options);
    const numFrames = frames.length;
    if (numFrames === 0) {
      return {
        data: new Float32Array(0),
        shape: layout === 'NTC' ? [1, 0, numMelBins] : [1, numMelBins, 0],
        numFrames: 0,
        numBins: numMelBins,
      };
    }

    const filterbank = AudioProcessor.createMelFilterbank(
      numMelBins,
      sampleRate,
      fftSize,
      minFreq,
      maxFreq
    );

    const totalElements = numFrames * numMelBins;
    const data = new Float32Array(totalElements);

    for (let f = 0; f < numFrames; f++) {
      const power = AudioProcessor.computePowerSpectrum(frames[f], fftSize);
      for (let m = 0; m < numMelBins; m++) {
        const filter = filterbank[m];
        let energy = 0;
        for (let k = 0; k < filter.length; k++) {
          energy += power[k] * filter[k];
        }
        const val = logMel ? Math.log(Math.max(energy, epsilon)) : energy;
        if (layout === 'NTC') {
          data[f * numMelBins + m] = val;
        } else {
          data[m * numFrames + f] = val;
        }
      }
    }

    const shape: [number, number, number] =
      layout === 'NTC' ? [1, numFrames, numMelBins] : [1, numMelBins, numFrames];

    return {
      data,
      shape,
      numFrames,
      numBins: numMelBins,
    };
  }
}
