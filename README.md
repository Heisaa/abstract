# Painting Reference

A local browser app for turning photos into oil-painting reference studies: large paintable value and color masses, brush-scale simplification, and selective edges. Filters preview with WebGPU, and export renders through the same pipeline so the file matches the preview.

## Filters

- **Analyze**
  - **Value**: posterized OKLab lightness in equal perceptual steps, with optional squint blur and kept color.
  - **Value Masses**: 2–5 connected value masses (notan). Thresholds are found automatically where the image's own values separate (multi-level Otsu), or set by hand; small specks merge into their surroundings.
  - **Edges**: flow-guided difference-of-Gaussians lines.
  - **Chroma**: color intensity as a heat or gray map.
  - **Warm / Cool**: areas classified as warm or cool.
- **Simplify**
  - **Kuwahara**: anisotropic Kuwahara that follows edge direction.
  - **SLIC**: OKLab superpixels, cleaned so each region is connected; neighboring patches of nearly the same color merge, so quiet areas become one large patch (a brush-size map).
  - **Color Check**: graph-based perceptual color fields, segmented in chroma-weighted OKLab so hue shifts (lips on skin, a head against foliage) stay separate.
  - **Palette**: chroma-weighted k-means palette extraction that keeps small saturated accents, or mapping to and checking against the Zorn palette gamut.
- **Render**
  - **Lost & Found**: broad simplified passages; edges with strong value contrast stay found, weak ones are lost, and the strongest keep sparse crisp accents.
  - **Flow Brush**: paint smeared along the smoothed flow field; strokes stop at color edges, and quiet areas get one consistent stroke direction.
  - **Adaptive Brush**: large brush cells in quiet areas, smaller ones where detail demands it; cells snap to color boundaries.
  - **Painter**: underpainting plus layered strokes from a large brush down to a small one. Finer layers only paint where the canvas still differs from the image, and strokes end at edges. Built-in styles range from tight (Sargent) to bold blocks.

## Studio tools

- Import by file picker, drag and drop, the Paste button, or Ctrl+V / ⌘V anywhere.
- Zoom (mouse wheel, toolbar) and pan (drag); double-click fits the image.
- Split compare slider, or hold Space to show the original.
- Grayscale, mirror, an N×N proportional grid with diagonals, and an edge overlay on top of any filter.
- Crop to panel ratios (8×10, 9×12, 11×14, 5×7, …).
- Eyedropper that reports value (0–10), chroma, and hue for the original and the filtered image.
- Built-in styles for the stroke filters, plus your own named presets per filter. Settings persist in localStorage.
- Export as PNG or JPEG, at preview size or high resolution (up to 3072 px on the long edge).

Keyboard shortcuts: Space (hold) shows the original · S split · G grayscale · M mirror · E edge overlay · I eyedropper · Esc stops picking.

## Architecture

- `src/filters/registry.ts`: one definition per filter (group, engine, parameters with ranges, units, and render-only flags). The sidebar, cache keys, preset validation, and resolution scaling are generated from it.
- `src/gpu/renderer.ts`: WebGPU renderer. All operations run through one serialized queue. Pipeline: source → filter stage (GPU passes, a worker job, or both) → cached filtered texture → composite pass (crop, mirror, split, grayscale, grid, overlay) → canvas or offscreen export texture.
- `src/gpu/shaders.ts`: WGSL. Each filter pass is its own entry point and pipeline; SLIC clustering is a compute shader.
- `src/cpu/*`: CPU filters (segmentation, value masses, palette, brush rendering), run in a Web Worker. Each filter has an `analyze` stage, cached on the parameters that affect it, and a cheap `render` stage, so recolor-only changes stay fast. While a slider is dragged, CPU filters preview at half resolution.

## Development

```sh
npm install
npm run dev
```

Use a WebGPU-capable browser. If WebGPU is unavailable, the app shows an in-app message.

## Deployment

Every push to `master` builds the app, runs the tests, and publishes `dist/` to GitHub Pages (`.github/workflows/deploy.yml`). One-time setup in the GitHub repository: Settings → Pages → Build and deployment → Source: **GitHub Actions**. Asset paths are relative (`base: './'`), so the site works at `https://<user>.github.io/<repo>/` under any repository name.

## Checks

```sh
npm run build   # type-check and production build
npm run lint
npm test        # Vitest unit tests for the CPU algorithms, registry, and view math
```
