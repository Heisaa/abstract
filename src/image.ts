// Image decoding. The preview works on a downscaled copy for interactive
// speed; the original file is kept so exports can be re-decoded at higher
// resolution.

export const PREVIEW_MAX_DIMENSION = 1280
export const EXPORT_MAX_DIMENSION = 3072

export const IMAGE_FILE_EXTENSIONS = ['.avif', '.bmp', '.gif', '.jpeg', '.jpg', '.jfif', '.png', '.webp']

export type DecodedPixels = {
  pixels: Uint8ClampedArray
  width: number
  height: number
}

export type LoadedImage = DecodedPixels & {
  name: string
  file: Blob
  sourceWidth: number
  sourceHeight: number
}

export function isSupportedImageFile(file: File) {
  if (file.type.startsWith('image/')) {
    return true
  }
  const lowerName = file.name.toLowerCase()
  return IMAGE_FILE_EXTENSIONS.some((extension) => lowerName.endsWith(extension))
}

export function fitWithin(width: number, height: number, maxDimension: number) {
  const scale = Math.min(1, maxDimension / Math.max(width, height))
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  }
}

async function decodeBitmap(file: Blob, width?: number, height?: number) {
  return createImageBitmap(file, {
    imageOrientation: 'from-image',
    premultiplyAlpha: 'none',
    ...(width && height ? { resizeWidth: width, resizeHeight: height, resizeQuality: 'high' as const } : {}),
  })
}

function bitmapPixels(bitmap: ImageBitmap): DecodedPixels {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Could not read image pixels.')
  }
  // Transparent areas read as white paper rather than black.
  context.fillStyle = '#ffffff'
  context.fillRect(0, 0, bitmap.width, bitmap.height)
  context.drawImage(bitmap, 0, 0)
  return { pixels: context.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, height: bitmap.height }
}

export async function decodeAtMaxDimension(file: Blob, maxDimension: number): Promise<DecodedPixels & { sourceWidth: number; sourceHeight: number }> {
  const source = await decodeBitmap(file)
  const sourceWidth = source.width
  const sourceHeight = source.height
  const target = fitWithin(sourceWidth, sourceHeight, maxDimension)
  let bitmap = source
  if (target.width !== sourceWidth || target.height !== sourceHeight) {
    source.close()
    bitmap = await decodeBitmap(file, target.width, target.height)
  }

  try {
    return { ...bitmapPixels(bitmap), sourceWidth, sourceHeight }
  } finally {
    bitmap.close()
  }
}

export async function loadImageFile(file: File): Promise<LoadedImage> {
  const decoded = await decodeAtMaxDimension(file, PREVIEW_MAX_DIMENSION)
  return { ...decoded, name: file.name, file }
}

// Long-edge size a high-res export will have for this image.
export function highResExportSize(image: Pick<LoadedImage, 'sourceWidth' | 'sourceHeight'>) {
  return fitWithin(image.sourceWidth, image.sourceHeight, EXPORT_MAX_DIMENSION)
}

export async function encodePixels(image: DecodedPixels, type: 'image/png' | 'image/jpeg', quality?: number) {
  const canvas = new OffscreenCanvas(image.width, image.height)
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('Could not encode the exported image.')
  }
  context.putImageData(new ImageData(new Uint8ClampedArray(image.pixels), image.width, image.height), 0, 0)
  return canvas.convertToBlob({ type, quality })
}

export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.click()
  // Give the browser time to start the download before releasing the URL.
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
}
