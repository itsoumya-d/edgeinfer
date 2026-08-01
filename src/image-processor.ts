// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com

export class ImageProcessor {
  static imageDataToFloat32Array(
    imageData: ImageData,
    options: { normalize?: boolean; layout?: 'NCHW' | 'NHWC' } = {}
  ): Float32Array {
    const { normalize = true, layout = 'NCHW' } = options;
    const { width, height, data } = imageData;
    const size = width * height;
    const float32Data = new Float32Array(size * 3);

    for (let i = 0; i < size; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];

      const nr = normalize ? r / 255 : r;
      const ng = normalize ? g / 255 : g;
      const nb = normalize ? b / 255 : b;

      if (layout === 'NCHW') {
        float32Data[i] = nr;
        float32Data[size + i] = ng;
        float32Data[2 * size + i] = nb;
      } else {
        float32Data[i * 3] = nr;
        float32Data[i * 3 + 1] = ng;
        float32Data[i * 3 + 2] = nb;
      }
    }
    return float32Data;
  }
}
