// Region utilities shared by the segmentation filters: union-find, connected
// components over a class map, and merging of regions below a minimum area.

export class DisjointSet {
  readonly parent: Int32Array
  readonly size: Int32Array

  constructor(count: number) {
    this.parent = new Int32Array(count)
    this.size = new Int32Array(count)
    for (let index = 0; index < count; index += 1) {
      this.parent[index] = index
      this.size[index] = 1
    }
  }

  find(index: number) {
    let root = index
    while (this.parent[root] !== root) {
      root = this.parent[root]
    }

    while (this.parent[index] !== root) {
      const next = this.parent[index]
      this.parent[index] = root
      index = next
    }

    return root
  }

  // Unions two roots; the larger set's root survives and is returned.
  unionRoots(rootA: number, rootB: number) {
    if (rootA === rootB) {
      return rootA
    }

    if (this.size[rootA] < this.size[rootB]) {
      const swap = rootA
      rootA = rootB
      rootB = swap
    }

    this.parent[rootB] = rootA
    this.size[rootA] += this.size[rootB]
    return rootA
  }
}

export type RegionMap = {
  labels: Int32Array
  count: number
  // Class value (band, palette index, superpixel id...) of each region.
  classOf: Int32Array
  sizes: Int32Array
}

// 4-connected components of equal class values.
export function connectedComponents(classes: ArrayLike<number>, width: number, height: number): RegionMap {
  const count = width * height
  const labels = new Int32Array(count).fill(-1)
  const stack = new Int32Array(count)
  const classList: number[] = []
  const sizeList: number[] = []
  let regionCount = 0

  for (let start = 0; start < count; start += 1) {
    if (labels[start] !== -1) {
      continue
    }

    const regionClass = classes[start]
    let top = 0
    let size = 0
    stack[top++] = start
    labels[start] = regionCount
    while (top > 0) {
      const index = stack[--top]
      size += 1
      const x = index % width
      if (x > 0 && labels[index - 1] === -1 && classes[index - 1] === regionClass) {
        labels[index - 1] = regionCount
        stack[top++] = index - 1
      }
      if (x + 1 < width && labels[index + 1] === -1 && classes[index + 1] === regionClass) {
        labels[index + 1] = regionCount
        stack[top++] = index + 1
      }
      if (index >= width && labels[index - width] === -1 && classes[index - width] === regionClass) {
        labels[index - width] = regionCount
        stack[top++] = index - width
      }
      if (index + width < count && labels[index + width] === -1 && classes[index + width] === regionClass) {
        labels[index + width] = regionCount
        stack[top++] = index + width
      }
    }

    classList.push(regionClass)
    sizeList.push(size)
    regionCount += 1
  }

  return {
    labels,
    count: regionCount,
    classOf: Int32Array.from(classList),
    sizes: Int32Array.from(sizeList),
  }
}

// Merges every region smaller than minSize into the neighbor it shares the
// longest border with. The surviving region keeps the class of the larger of
// the two, so a small speck adopts its surroundings.
export function mergeSmallRegions(regions: RegionMap, width: number, height: number, minSize: number): RegionMap {
  const { labels, count, classOf } = regions
  if (minSize <= 1 || count <= 1) {
    return regions
  }

  const sets = new DisjointSet(count)
  for (let region = 0; region < count; region += 1) {
    sets.size[region] = regions.sizes[region]
  }

  const rootClass = Int32Array.from(classOf)

  for (let pass = 0; pass < 12; pass += 1) {
    const borders = new Map<number, Map<number, number>>()
    const addBorder = (small: number, neighbor: number) => {
      let counts = borders.get(small)
      if (!counts) {
        counts = new Map()
        borders.set(small, counts)
      }
      counts.set(neighbor, (counts.get(neighbor) ?? 0) + 1)
    }

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = x + y * width
        const root = sets.find(labels[index])
        const rootSmall = sets.size[root] < minSize
        if (x + 1 < width) {
          const right = sets.find(labels[index + 1])
          if (right !== root) {
            if (rootSmall) addBorder(root, right)
            if (sets.size[right] < minSize) addBorder(right, root)
          }
        }
        if (y + 1 < height) {
          const down = sets.find(labels[index + width])
          if (down !== root) {
            if (rootSmall) addBorder(root, down)
            if (sets.size[down] < minSize) addBorder(down, root)
          }
        }
      }
    }

    if (borders.size === 0) {
      break
    }

    let merged = false
    // Smallest regions first so specks are absorbed before their neighbors move.
    const smallRoots = Array.from(borders.keys()).sort((a, b) => sets.size[a] - sets.size[b])
    for (const small of smallRoots) {
      const root = sets.find(small)
      if (root !== small || sets.size[root] >= minSize) {
        continue
      }

      let best = -1
      let bestCount = -1
      for (const [neighbor, borderCount] of borders.get(small)!) {
        const neighborRoot = sets.find(neighbor)
        if (neighborRoot === root) {
          continue
        }
        if (borderCount > bestCount || (borderCount === bestCount && sets.size[neighborRoot] > sets.size[best])) {
          best = neighborRoot
          bestCount = borderCount
        }
      }

      if (best < 0) {
        continue
      }

      const winnerClass = sets.size[best] >= sets.size[root] ? rootClass[best] : rootClass[root]
      const survivor = sets.unionRoots(root, best)
      rootClass[survivor] = winnerClass
      merged = true
    }

    if (!merged) {
      break
    }
  }

  return compactRegions(labels, count, sets, rootClass, width * height)
}

function compactRegions(labels: Int32Array, count: number, sets: DisjointSet, rootClass: Int32Array, pixelCount: number): RegionMap {
  const remap = new Int32Array(count).fill(-1)
  const classList: number[] = []
  const sizeList: number[] = []
  const out = new Int32Array(pixelCount)

  for (let index = 0; index < pixelCount; index += 1) {
    const root = sets.find(labels[index])
    let label = remap[root]
    if (label === -1) {
      label = classList.length
      remap[root] = label
      classList.push(rootClass[root])
      sizeList.push(0)
    }
    sizeList[label] += 1
    out[index] = label
  }

  return {
    labels: out,
    count: classList.length,
    classOf: Int32Array.from(classList),
    sizes: Int32Array.from(sizeList),
  }
}

// Marks pixels on region boundaries, dilated to the given thickness.
export function boundaryMask(labels: Int32Array, width: number, height: number, thickness: number) {
  const count = width * height
  let boundary = new Uint8Array(count)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + y * width
      const label = labels[index]
      if ((x + 1 < width && labels[index + 1] !== label) || (y + 1 < height && labels[index + width] !== label)) {
        boundary[index] = 1
      }
    }
  }

  for (let pass = 1; pass < thickness; pass += 1) {
    const dilated = new Uint8Array(boundary)
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = x + y * width
        if (
          !boundary[index] &&
          ((x > 0 && boundary[index - 1]) ||
            (x + 1 < width && boundary[index + 1]) ||
            (y > 0 && boundary[index - width]) ||
            (y + 1 < height && boundary[index + width]))
        ) {
          dilated[index] = 1
        }
      }
    }
    boundary = dilated
  }

  return boundary
}
