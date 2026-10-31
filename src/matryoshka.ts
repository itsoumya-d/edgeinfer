// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath
//
// Browser-local two-stage Matryoshka retrieval.
// Stage 1: INT8 truncated vectors (default 64 dims) over the full index.
// Stage 2: FP32 re-ranking of the top candidates at the full dimension.

export interface VectorRecord {
  id: string;
  text?: string;
  embedding: Float32Array;
  metadata?: Record<string, unknown>;
}

export interface SearchResult {
  id: string;
  text?: string;
  metadata?: Record<string, unknown>;
  score: number;
  coarseScore: number;
}

export interface VectorStore {
  add(records: VectorRecord[]): Promise<void>;
  getAll(): Promise<VectorRecord[]>;
  clear(): Promise<void>;
  count(): Promise<number>;
  close?(): void;
}

export interface MatryoshkaOptions {
  dimensions?: number;
  coarseDimension?: number;
  refineTopK?: number;
  store?: VectorStore;
}

export interface SearchOptions {
  topK?: number;
  refineTopK?: number;
  coarseDimension?: number;
  refine?: boolean;
}

export function l2Normalize(vector: Float32Array, dimension = vector.length): Float32Array {
  const output = new Float32Array(dimension);
  if (dimension > vector.length) {
    throw new Error(`Cannot normalize ${dimension} dims from a ${vector.length}-dim vector`);
  }
  let norm = 0;
  for (let i = 0; i < dimension; i++) norm += vector[i] * vector[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return output;
  for (let i = 0; i < dimension; i++) output[i] = vector[i] / norm;
  return output;
}

export function truncateAndNormalize(vector: Float32Array, dimension: number): Float32Array {
  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error("dimension must be a positive integer");
  }
  return l2Normalize(vector, dimension);
}

export function quantizeInt8(
  vector: Float32Array,
  dimension = vector.length
): { data: Int8Array; scale: number } {
  let maxAbs = 0;
  for (let i = 0; i < dimension; i++) {
    const value = Math.abs(vector[i]);
    if (value > maxAbs) maxAbs = value;
  }
  const scale = maxAbs > 0 ? maxAbs / 127 : 1;
  const data = new Int8Array(dimension);
  for (let i = 0; i < dimension; i++) {
    const value = Math.round(vector[i] / scale);
    data[i] = value > 127 ? 127 : value < -127 ? -127 : value;
  }
  return { data, scale };
}

export function dequantizeInt8(data: Int8Array, scale: number): Float32Array {
  const output = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) output[i] = data[i] * scale;
  return output;
}

function dot(a: Float32Array | Int8Array, b: Float32Array | Int8Array, dimension: number): number {
  let sum = 0;
  for (let i = 0; i < dimension; i++) sum += a[i] * b[i];
  return sum;
}

class TopKHeap {
  private items: Array<{ index: number; score: number }> = [];

  constructor(private readonly k: number) {
    if (k < 1) throw new Error("top-k must be positive");
  }

  push(index: number, score: number): void {
    if (this.items.length < this.k) {
      this.items.push({ index, score });
      if (this.items.length === this.k) this.heapify();
      return;
    }
    if (score > this.items[0].score) {
      this.items[0] = { index, score };
      this.siftDown(0);
    }
  }

  private heapify(): void {
    for (let i = Math.floor(this.items.length / 2) - 1; i >= 0; i--) this.siftDown(i);
  }

  private siftDown(start: number): void {
    let index = start;
    const length = this.items.length;
    for (;;) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < length && this.items[left].score < this.items[smallest].score) smallest = left;
      if (right < length && this.items[right].score < this.items[smallest].score) smallest = right;
      if (smallest === index) break;
      [this.items[index], this.items[smallest]] = [this.items[smallest], this.items[index]];
      index = smallest;
    }
  }

  values(): Array<{ index: number; score: number }> {
    return [...this.items].sort((a, b) => b.score - a.score);
  }
}

export class MemoryVectorStore implements VectorStore {
  private records: VectorRecord[] = [];

  async add(records: VectorRecord[]): Promise<void> {
    this.records.push(...records);
  }

  async getAll(): Promise<VectorRecord[]> {
    return this.records.slice();
  }

  async clear(): Promise<void> {
    this.records = [];
  }

  async count(): Promise<number> {
    return this.records.length;
  }
}

interface StoredVectorRecord {
  id: string;
  text?: string;
  metadata?: Record<string, unknown>;
  embedding: ArrayBuffer;
}

export class IndexedDbVectorStore implements VectorStore {
  private dbPromise: Promise<IDBDatabase> | null = null;

  constructor(
    private readonly dbName = "edgeinfer-matryoshka",
    private readonly storeName = "vectors"
  ) {}

  private open(): Promise<IDBDatabase> {
    if (this.dbPromise) return this.dbPromise;
    if (typeof indexedDB === "undefined") {
      return Promise.reject(new Error("IndexedDB is unavailable in this environment"));
    }
    this.dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: "id" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
    });
    return this.dbPromise;
  }

  async add(records: VectorRecord[]): Promise<void> {
    const db = await this.open();
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readwrite");
      const store = transaction.objectStore(this.storeName);
      for (const record of records) {
        const stored: StoredVectorRecord = {
          id: record.id,
          text: record.text,
          metadata: record.metadata,
          embedding: record.embedding.slice().buffer,
        };
        store.put(stored);
      }
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB write failed"));
      transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB write aborted"));
    });
  }

  async getAll(): Promise<VectorRecord[]> {
    const db = await this.open();
    return new Promise<VectorRecord[]>((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.getAll();
      request.onsuccess = () => {
        const records = (request.result as StoredVectorRecord[]).map((stored) => ({
          id: stored.id,
          text: stored.text,
          metadata: stored.metadata,
          embedding: new Float32Array(stored.embedding),
        }));
        resolve(records);
      };
      request.onerror = () => reject(request.error ?? new Error("IndexedDB read failed"));
    });
  }

  async clear(): Promise<void> {
    const db = await this.open();
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readwrite");
      transaction.objectStore(this.storeName).clear();
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB clear failed"));
    });
  }

  async count(): Promise<number> {
    const db = await this.open();
    return new Promise<number>((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readonly");
      const request = transaction.objectStore(this.storeName).count();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB count failed"));
    });
  }

  close(): void {
    void this.dbPromise?.then((db) => db.close());
    this.dbPromise = null;
  }
}

export class MatryoshkaIndex {
  readonly dimensions: number;
  readonly coarseDimension: number;
  readonly refineTopK: number;
  private store?: VectorStore;
  private ids: string[] = [];
  private texts: Array<string | undefined> = [];
  private metadata: Array<Record<string, unknown> | undefined> = [];
  private full: Float32Array = new Float32Array(0);
  private coarse: Int8Array = new Int8Array(0);
  private scales: Float32Array = new Float32Array(0);
  private size = 0;

  constructor(options: MatryoshkaOptions = {}) {
    this.dimensions = options.dimensions ?? 768;
    this.coarseDimension = options.coarseDimension ?? 64;
    this.refineTopK = options.refineTopK ?? 50;
    this.store = options.store;
    if (this.coarseDimension > this.dimensions) {
      throw new Error("coarseDimension cannot exceed dimensions");
    }
  }

  get count(): number {
    return this.size;
  }

  async build(records: VectorRecord[], { persist = true } = {}): Promise<void> {
    if (records.length === 0) throw new Error("At least one record is required");
    for (const record of records) {
      if (record.embedding.length !== this.dimensions) {
        throw new Error(
          `Record ${record.id} has ${record.embedding.length} dims; expected ${this.dimensions}`
        );
      }
    }

    this.ids = new Array(records.length);
    this.texts = new Array(records.length);
    this.metadata = new Array(records.length);
    this.full = new Float32Array(records.length * this.dimensions);
    this.coarse = new Int8Array(records.length * this.coarseDimension);
    this.scales = new Float32Array(records.length);
    this.size = records.length;

    for (let row = 0; row < records.length; row++) {
      const record = records[row];
      const normalized = l2Normalize(record.embedding, this.dimensions);
      this.ids[row] = record.id;
      this.texts[row] = record.text;
      this.metadata[row] = record.metadata;
      this.full.set(normalized, row * this.dimensions);

      const quantized = quantizeInt8(normalized, this.coarseDimension);
      this.coarse.set(quantized.data, row * this.coarseDimension);
      this.scales[row] = quantized.scale;
    }

    if (persist && this.store) {
      await this.store.clear();
      const normalizedRecords = records.map((record, row) => ({
        ...record,
        embedding: this.full.slice(row * this.dimensions, (row + 1) * this.dimensions),
      }));
      await this.store.add(normalizedRecords);
    }
  }

  static async load(store: VectorStore, options: MatryoshkaOptions = {}): Promise<MatryoshkaIndex> {
    const index = new MatryoshkaIndex({ ...options, store });
    const records = await store.getAll();
    await index.build(records, { persist: false });
    return index;
  }

  search(query: Float32Array, options: SearchOptions = {}): SearchResult[] {
    if (this.size === 0) throw new Error("Index is empty; call build() first");
    const topK = options.topK ?? 10;
    const refine = options.refine ?? true;
    const coarseDimension = options.coarseDimension ?? this.coarseDimension;
    if (coarseDimension > this.coarseDimension) {
      throw new Error(
        `coarseDimension ${coarseDimension} exceeds the indexed stride ${this.coarseDimension}`
      );
    }
    const queryFull = l2Normalize(query, this.dimensions);
    const queryCoarse = quantizeInt8(queryFull, coarseDimension);
    const refineTopK = Math.max(
      topK,
      Math.min(this.refineTopK, options.refineTopK ?? this.refineTopK, this.size)
    );

    const heap = new TopKHeap(refineTopK);
    for (let row = 0; row < this.size; row++) {
      const offset = row * this.coarseDimension;
      let sum = 0;
      for (let i = 0; i < coarseDimension; i++) {
        sum += queryCoarse.data[i] * this.coarse[offset + i];
      }
      heap.push(row, sum * this.scales[row]);
    }

    let candidates: Array<{ index: number; score: number; coarseScore: number }> =
      heap.values().map((candidate) => ({ ...candidate, coarseScore: candidate.score }));
    if (refine) {
      candidates = candidates
        .map((candidate) => ({
          ...candidate,
          score: dot(
            queryFull,
            this.full.subarray(
              candidate.index * this.dimensions,
              (candidate.index + 1) * this.dimensions
            ),
            this.dimensions
          ),
        }))
        .sort((a, b) => b.score - a.score);
    }

    return candidates.slice(0, topK).map((candidate) => ({
      id: this.ids[candidate.index],
      text: this.texts[candidate.index],
      metadata: this.metadata[candidate.index],
      score: candidate.score,
      coarseScore: candidate.coarseScore,
    }));
  }

  exactSearchAtDimension(
    query: Float32Array,
    dimension: number,
    topK = 10
  ): Array<{ index: number; score: number }> {
    if (dimension > this.dimensions) throw new Error("dimension exceeds index dimensions");
    const queryNormalized = truncateAndNormalize(query, dimension);
    const scored: Array<{ index: number; score: number }> = [];
    for (let row = 0; row < this.size; row++) {
      const vector = truncateAndNormalize(
        this.full.subarray(row * this.dimensions, (row + 1) * this.dimensions),
        dimension
      );
      scored.push({ index: row, score: dot(queryNormalized, vector, dimension) });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }

  recallAtK(
    queries: Float32Array[],
    groundTruth: string[][],
    dimension: number,
    topK = 10
  ): number {
    let hits = 0;
    let total = 0;
    for (let q = 0; q < queries.length; q++) {
      const expected = new Set(groundTruth[q].slice(0, topK));
      const candidates = this.exactSearchAtDimension(queries[q], dimension, topK);
      for (const candidate of candidates) {
        if (expected.has(this.ids[candidate.index])) hits++;
      }
      total += Math.min(topK, expected.size);
    }
    return total === 0 ? 0 : hits / total;
  }

  get idList(): readonly string[] {
    return this.ids;
  }
}

export type EmbedFn = (text: string) => Promise<Float32Array> | Float32Array;

export async function embedMatryoshka(
  text: string,
  dimension: number,
  embed: EmbedFn
): Promise<Float32Array> {
  const full = await embed(text);
  return truncateAndNormalize(full, dimension);
}

export function createMatryoshkaEmbedder(embed: EmbedFn) {
  return (text: string, dimension = 768) => embedMatryoshka(text, dimension, embed);
}
