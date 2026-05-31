# Painting Reference

A local browser app for turning imported images into oil-painting reference studies with fast WebGPU shader previews.

## Current Features

- Import images by file picker or drag and drop.
- Preview filters with WebGPU/WGSL.
- Filters are grouped by workflow stage:
  - **Analyze** — Value (posterize) and Edges.
  - **Simplify** — Kuwahara, SLIC, and Color Check (perceptual color-field segmentation).
  - **Render** — Flow Brush, Adaptive Brush, and Painter (block/smooth base with edge- or color-driven detail).
- Toggle between original and filtered views.
- Export the current filtered image as PNG or JPEG.

## Development

```sh
npm install
npm run dev
```

Use a WebGPU-capable browser. If WebGPU is unavailable, the app shows an in-app compatibility message.

## Build

```sh
npm run build
```
