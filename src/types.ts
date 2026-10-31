// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

export interface ModelOptions {
  executionProviders?: string[];
  cacheName?: string;
  forceDownload?: boolean;
}

export interface ClassificationResult {
  label: string;
  score: number;
}

export interface Detection {
  bbox: [number, number, number, number];
  label: string;
  score: number;
}
