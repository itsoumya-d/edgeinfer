// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com | +91 7031648617

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
