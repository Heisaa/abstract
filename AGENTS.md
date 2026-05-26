# Project Notes

- Build a local image-reference tool for oil painting studies.
- Recommended stack: Vite, TypeScript, React, WebGPU/WGSL shaders.
- Start as a browser-local app; package later with Tauri 2 if a desktop shell is needed.
- Core workflow: import image, choose one filter, adjust relevant parameters, preview quickly, export filtered image.
- Initial filters: Kuwahara, posterize/value study, edge overlay, blur/sharpen, and later superpixel abstraction.
- Filter design should prioritize painter study over cosmetic effects: identify large paintable color/value masses, approximate brush-scale fields, and preserve only selective detail where structure matters.
- Highest-value additions: superpixel brush map, adaptive brush detail, lost-and-found edge control, palette-field simplification, and value-mass grouping.
- For Sargent-like studies, favor broad simplified passages with sparse crisp accents; expose brush size, detail threshold, merge tolerance, and edge/detail strength as adjustable controls.
- Prefer GPU shader preview and export from the same rendering path so output matches the preview.
- Before installing dependencies or scaffolding, search the web for the latest official package versions, setup commands, and installation instructions.
- Run npm run build or tsc --noEmit to verify result
- Do not run npm run dev
