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
