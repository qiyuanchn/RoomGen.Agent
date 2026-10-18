// Walk-mode geometry for the RoomGen.Agent 3D viewer.
//
// The GLB is the ground truth: `createWalkIndex` collects world-space GLB
// triangles into two tiny spatial grids, one for surfaces that can be stood on
// (up-facing floors, stair treads, landings, ramps) and one for the faces a
// body can bump into (walls, furniture sides, stair risers).  The walker asks
// the index for the surface under the next position - a downward ray, not a
// fixed eye height - and refuses a step when the only surface there is far
// below (a stairwell opening, a room edge) or when a wall/furniture face is in
// the way.
//
// This module has no three.js dependency so the geometry can be exercised by
// `node tests/walk_geometry.test.mjs`; viewer.js feeds it the loaded GLB and
// adds the eye height on top of the returned surface.

export const WALK_DEFAULTS = Object.freeze({
  maxStepUp: 0.34,     // a stair riser / ramp lip the walker can climb
  maxDrop: 0.6,        // one stair tread down; a slab opening is far deeper
  bodyRadius: 0.24,
  bodyHeight: 1.7,     // head clearance above the feet
  stepClearance: 0.35, // a face below this is the step the support index climbs
  gapReach: 0.15,      // a seam/crack shorter than this is stepped across
  cellSize: 1.0,
  // placement search, tried in rings around a room centre or camera pose
  searchRings: Object.freeze([0.35, 0.6, 0.9, 1.2, 1.5]),
});

const EPS = 1e-7;
const SUPPORT_NORMAL_Y = 0.3;   // floors, treads, gentle ramps
const OBSTACLE_NORMAL_Y = 0.9;  // anything not close to horizontal

// --------------------------------------------------------------------------- //
// triangle store + grid
// --------------------------------------------------------------------------- //
function createStore() {
  return {
    data: new Float32Array(9 * 2048),
    count: 0,
    yMin: new Float32Array(2048),
    yMax: new Float32Array(2048),
    cells: new Map(),
  };
}

function ensureCapacity(store, needed) {
  if (needed <= store.data.length / 9) {
    return;
  }
  let capacity = store.data.length / 9;
  while (capacity < needed) {
    capacity *= 2;
  }
  const nextData = new Float32Array(capacity * 9);
  nextData.set(store.data.subarray(0, store.count * 9));
  store.data = nextData;
  const nextMin = new Float32Array(capacity);
  nextMin.set(store.yMin.subarray(0, store.count));
  store.yMin = nextMin;
  const nextMax = new Float32Array(capacity);
  nextMax.set(store.yMax.subarray(0, store.count));
  store.yMax = nextMax;
}

function cellKey(cx, cz) {
  return `${cx},${cz}`;
}

function storeTriangle(store, cellSize, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const id = store.count;
  ensureCapacity(store, id + 1);
  const base = id * 9;
  store.data[base] = ax;
  store.data[base + 1] = ay;
  store.data[base + 2] = az;
  store.data[base + 3] = bx;
  store.data[base + 4] = by;
  store.data[base + 5] = bz;
  store.data[base + 6] = cx;
  store.data[base + 7] = cy;
  store.data[base + 8] = cz;
  store.count = id + 1;

  const yMin = Math.min(ay, by, cy);
  const yMax = Math.max(ay, by, cy);
  const xMin = Math.min(ax, bx, cx);
  const xMax = Math.max(ax, bx, cx);
  const zMin = Math.min(az, bz, cz);
  const zMax = Math.max(az, bz, cz);
  store.yMin[id] = yMin;
  store.yMax[id] = yMax;

  const cx0 = Math.floor(xMin / cellSize);
  const cx1 = Math.floor(xMax / cellSize);
  const cz0 = Math.floor(zMin / cellSize);
  const cz1 = Math.floor(zMax / cellSize);
  for (let ix = cx0; ix <= cx1; ix += 1) {
    for (let iz = cz0; iz <= cz1; iz += 1) {
      const key = cellKey(ix, iz);
      let bucket = store.cells.get(key);
      if (!bucket) {
        bucket = [];
        store.cells.set(key, bucket);
      }
      bucket.push(id);
    }
  }
}

function cellTriangles(store, cellSize, x, z) {
  return store.cells.get(cellKey(Math.floor(x / cellSize), Math.floor(z / cellSize))) || null;
}

// --------------------------------------------------------------------------- //
// ray / triangle
// --------------------------------------------------------------------------- //
function rayTriangle(ox, oy, oz, dx, dy, dz,
                     ax, ay, az, bx, by, bz, cx, cy, cz) {
  const e1x = bx - ax;
  const e1y = by - ay;
  const e1z = bz - az;
  const e2x = cx - ax;
  const e2y = cy - ay;
  const e2z = cz - az;

  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < EPS) {
    return -1;
  }
  const inv = 1 / det;
  const tx = ox - ax;
  const ty = oy - ay;
  const tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u < -1e-6 || u > 1 + 1e-6) {
    return -1;
  }

  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < -1e-6 || u + v > 1 + 1e-6) {
    return -1;
  }
  return (e2x * qx + e2y * qy + e2z * qz) * inv;
}

function rayDown(store, id, x, z, originY, minY) {
  const base = id * 9;
  if (store.yMax[id] < minY - EPS || store.yMin[id] > originY + EPS) {
    return -1;
  }
  const data = store.data;
  const t = rayTriangle(
    x, originY, z,
    0, -1, 0,
    data[base], data[base + 1], data[base + 2],
    data[base + 3], data[base + 4], data[base + 5],
    data[base + 6], data[base + 7], data[base + 8],
  );
  if (t < -EPS) {
    return -1;
  }
  return t;
}

// 2D helpers: the body is a vertical cylinder swept along the move, so the
// horizontal distance from that segment to a triangle's XZ projection decides
// whether a face is in the way.  This sees thin panels (a table top at waist
// height) that fixed-height horizontal rays slip between.
function pointSegmentDistance2(px, pz, ax, az, bx, bz) {
  const dx = bx - ax;
  const dz = bz - az;
  const lengthSq = dx * dx + dz * dz;
  if (lengthSq <= EPS) {
    return Math.hypot(px - ax, pz - az);
  }
  let t = ((px - ax) * dx + (pz - az) * dz) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + dx * t), pz - (az + dz * t));
}

function orientation2(ax, az, bx, bz, cx, cz) {
  return (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
}

function segmentsIntersect2(ax, az, bx, bz, cx, cz, dx, dz) {
  const o1 = orientation2(ax, az, bx, bz, cx, cz);
  const o2 = orientation2(ax, az, bx, bz, dx, dz);
  const o3 = orientation2(cx, cz, dx, dz, ax, az);
  const o4 = orientation2(cx, cz, dx, dz, bx, bz);
  return ((o1 > EPS && o2 < -EPS) || (o1 < -EPS && o2 > EPS))
    && ((o3 > EPS && o4 < -EPS) || (o3 < -EPS && o4 > EPS));
}

function segmentDistance2(ax, az, bx, bz, cx, cz, dx, dz) {
  if (segmentsIntersect2(ax, az, bx, bz, cx, cz, dx, dz)) {
    return 0;
  }
  return Math.min(
    pointSegmentDistance2(ax, az, cx, cz, dx, dz),
    pointSegmentDistance2(bx, bz, cx, cz, dx, dz),
    pointSegmentDistance2(cx, cz, ax, az, bx, bz),
    pointSegmentDistance2(dx, dz, ax, az, bx, bz),
  );
}

function pointInTriangle2(px, pz, ax, az, bx, bz, cx, cz) {
  const area = orientation2(ax, az, bx, bz, cx, cz);
  if (Math.abs(area) <= EPS) {
    return false;  // a wall seen from above is a line, its edges cover it
  }
  const s0 = orientation2(px, pz, ax, az, bx, bz);
  const s1 = orientation2(px, pz, bx, bz, cx, cz);
  const s2 = orientation2(px, pz, cx, cz, ax, az);
  const hasNeg = s0 < -EPS || s1 < -EPS || s2 < -EPS;
  const hasPos = s0 > EPS || s1 > EPS || s2 > EPS;
  return !(hasNeg && hasPos);
}

function segmentTriangleDistance2(x0, z0, x1, z1,
                                  ax, az, bx, bz, cx, cz) {
  let best = Math.min(
    segmentDistance2(x0, z0, x1, z1, ax, az, bx, bz),
    segmentDistance2(x0, z0, x1, z1, bx, bz, cx, cz),
    segmentDistance2(x0, z0, x1, z1, cx, cz, ax, az),
  );
  if (pointInTriangle2(x0, z0, ax, az, bx, bz, cx, cz)
      || pointInTriangle2(x1, z1, ax, az, bx, bz, cx, cz)) {
    best = 0;
  }
  return best;
}

// --------------------------------------------------------------------------- //
// public API
// --------------------------------------------------------------------------- //
export function createWalkIndex(options = {}) {
  return {
    cellSize: Number.isFinite(options.cellSize) ? options.cellSize : WALK_DEFAULTS.cellSize,
    support: createStore(),
    obstacle: createStore(),
    triangles: 0,
  };
}

/**
 * Add one world-space GLB triangle.  Up-facing faces join the support grid,
 * steep faces join the obstacle grid; a slope may belong to both.
 */
export function addTriangle(index, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const e1x = bx - ax;
  const e1y = by - ay;
  const e1z = bz - az;
  const e2x = cx - ax;
  const e2y = cy - ay;
  const e2z = cz - az;
  const nx = e1y * e2z - e1z * e2y;
  const ny = e1z * e2x - e1x * e2z;
  const nz = e1x * e2y - e1y * e2x;
  const length = Math.hypot(nx, ny, nz);
  if (length < EPS) {
    return false;
  }
  const normalY = ny / length;
  let stored = false;
  if (normalY > SUPPORT_NORMAL_Y) {
    storeTriangle(index.support, index.cellSize, ax, ay, az, bx, by, bz, cx, cy, cz);
    stored = true;
  }
  if (Math.abs(normalY) < OBSTACLE_NORMAL_Y) {
    storeTriangle(index.obstacle, index.cellSize, ax, ay, az, bx, by, bz, cx, cy, cz);
    stored = true;
  }
  if (stored) {
    index.triangles += 1;
  }
  return stored;
}

function optionsWithDefaults(options) {
  const merged = { ...WALK_DEFAULTS, ...(options || {}) };
  merged.maxStepUp = Number.isFinite(merged.maxStepUp) ? merged.maxStepUp : WALK_DEFAULTS.maxStepUp;
  merged.maxDrop = Number.isFinite(merged.maxDrop) ? merged.maxDrop : WALK_DEFAULTS.maxDrop;
  merged.bodyRadius = Number.isFinite(merged.bodyRadius) ? merged.bodyRadius : WALK_DEFAULTS.bodyRadius;
  merged.bodyHeight = Number.isFinite(merged.bodyHeight) ? merged.bodyHeight : WALK_DEFAULTS.bodyHeight;
  merged.stepClearance = Number.isFinite(merged.stepClearance)
    ? merged.stepClearance : WALK_DEFAULTS.stepClearance;
  merged.gapReach = Number.isFinite(merged.gapReach) ? merged.gapReach : WALK_DEFAULTS.gapReach;
  if (!Array.isArray(merged.searchRings) || !merged.searchRings.length) {
    merged.searchRings = WALK_DEFAULTS.searchRings;
  }
  return merged;
}

/**
 * Highest walkable surface at (x, z) inside the step window below the current
 * feet.  Returns `{ y }` or `null` when the only surface is too far down (a
 * stairwell opening) or too far up (a table, a wall top).
 */
export function supportAt(index, x, z, feetY, options = {}) {
  const opt = optionsWithDefaults(options);
  const originY = feetY + opt.maxStepUp;
  const minY = feetY - opt.maxDrop;
  const bucket = cellTriangles(index.support, index.cellSize, x, z);
  if (!bucket) {
    return null;
  }
  let best = null;
  for (const id of bucket) {
    const t = rayDown(index.support, id, x, z, originY, minY);
    if (t < 0) {
      continue;
    }
    const y = originY - t;
    if (y < minY - 1e-6 || y > originY + 1e-6) {
      continue;
    }
    if (best === null || y > best.y) {
      best = { y };
    }
  }
  return best;
}

/**
 * Highest surface under (x, z) between `fromY` and `minY`; used when the walker
 * is placed (mode switch, floor jump, room jump) rather than stepped.
 */
export function supportBelow(index, x, z, fromY, minY) {
  const bucket = cellTriangles(index.support, index.cellSize, x, z);
  if (!bucket) {
    return null;
  }
  let best = null;
  for (const id of bucket) {
    const t = rayDown(index.support, id, x, z, fromY, minY);
    if (t < 0) {
      continue;
    }
    const y = fromY - t;
    if (y < minY - 1e-6 || y > fromY + 1e-6) {
      continue;
    }
    if (best === null || y > best.y) {
      best = { y };
    }
  }
  return best;
}

function highestSupportBetween(index, x, z, low, high) {
  const bucket = cellTriangles(index.support, index.cellSize, x, z);
  if (!bucket) {
    return null;
  }
  let best = null;
  for (const id of bucket) {
    const t = rayDown(index.support, id, x, z, high, low);
    if (t < 0) {
      continue;
    }
    const y = high - t;
    if (y < low - 1e-6 || y > high + 1e-6) {
      continue;
    }
    if (best === null || y > best) {
      best = y;
    }
  }
  return best;
}

/**
 * True when a body of `bodyRadius` moving from (x0, z0) to (x1, z1) with its
 * feet at `feetY` would touch a wall, furniture side or stair riser.  The body
 * is a vertical cylinder swept along the move: a face blocks when its XZ
 * projection is within `bodyRadius` of the move and its height overlaps
 * `feetY + stepClearance .. feetY + bodyHeight`.  The clearance is what lets
 * the walker climb a riser: the step under the destination feet is below it.
 *
 * An up-facing surface directly above the body (a table top, a shelf, a stair
 * flight) also blocks: the walker cannot stand inside furniture whose top is
 * above the step window even when its sides are further than `bodyRadius`.
 */
export function blocked(index, x0, z0, x1, z1, feetY, options = {}) {
  const opt = optionsWithDefaults(options);
  if (Math.hypot(x1 - x0, z1 - z0) <= EPS) {
    return false;
  }
  const overheadLow = feetY + opt.maxStepUp + 1e-3;
  const overheadHigh = feetY + opt.bodyHeight;
  if (highestSupportBetween(index, x0, z0, overheadLow, overheadHigh) !== null
      || highestSupportBetween(index, x1, z1, overheadLow, overheadHigh) !== null) {
    return true;
  }
  const low = feetY + opt.stepClearance;
  const high = feetY + opt.bodyHeight;
  const radius = opt.bodyRadius;
  const cellSize = index.cellSize;
  const store = index.obstacle;
  const data = store.data;
  const cx0 = Math.floor((Math.min(x0, x1) - radius) / cellSize);
  const cx1 = Math.floor((Math.max(x0, x1) + radius) / cellSize);
  const cz0 = Math.floor((Math.min(z0, z1) - radius) / cellSize);
  const cz1 = Math.floor((Math.max(z0, z1) + radius) / cellSize);
  const seen = new Set();
  for (let cx = cx0; cx <= cx1; cx += 1) {
    for (let cz = cz0; cz <= cz1; cz += 1) {
      const bucket = store.cells.get(cellKey(cx, cz));
      if (!bucket) {
        continue;
      }
      for (const id of bucket) {
        if (seen.has(id)) {
          continue;
        }
        seen.add(id);
        if (store.yMax[id] <= low + EPS || store.yMin[id] >= high - EPS) {
          continue;
        }
        const base = id * 9;
        const distance = segmentTriangleDistance2(
          x0, z0, x1, z1,
          data[base], data[base + 2],
          data[base + 3], data[base + 5],
          data[base + 6], data[base + 8],
        );
        if (distance <= radius) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * A standing spot on the same storey near (x, z).
 *
 * Room centres are bounding-box centres and camera poses may sit inside a
 * wardrobe or on a slab opening.  The first candidate is the requested point;
 * when it is not free (or has no surface within one step) rings of candidates
 * are tried.  Returns `{ x, z, feetY }` or `null`.
 */
export function findStandingSpot(index, x, z, feetY, options = {}) {
  const opt = optionsWithDefaults(options);
  const wanted = typeof opt.acceptPoint === 'function' ? opt.acceptPoint : null;
  const free = (px, pz, feet) => {
    if (blocked(index, px, pz, px + 0.06, pz, feet, opt)
        || blocked(index, px, pz, px - 0.06, pz, feet, opt)
        || blocked(index, px, pz, px, pz + 0.06, feet, opt)
        || blocked(index, px, pz, px, pz - 0.06, feet, opt)) {
      return false;
    }
    return true;
  };
  const surfaceAt = (px, pz) => {
    const surface = supportBelow(index, px, pz, feetY + opt.maxStepUp, feetY - opt.maxDrop);
    if (!surface) {
      return null;
    }
    if (typeof opt.acceptSupport === 'function' && !opt.acceptSupport(surface.y)) {
      return null;
    }
    return surface.y;
  };
  const baseY = surfaceAt(x, z);
  if ((!wanted || wanted(x, z)) && baseY !== null && free(x, z, baseY)) {
    return { x, z, feetY: baseY };
  }
  for (const ring of opt.searchRings) {
    for (let step = 0; step < 8; step += 1) {
      const angle = (Math.PI / 4) * step;
      const px = x + Math.cos(angle) * ring;
      const pz = z + Math.sin(angle) * ring;
      if (wanted && !wanted(px, pz)) {
        continue;
      }
      const surfaceY = surfaceAt(px, pz);
      if (surfaceY === null) {
        continue;
      }
      if (free(px, pz, surfaceY)) {
        return { x: px, z: pz, feetY: surfaceY };
      }
    }
  }
  return null;
}

/**
 * One walk step: try the full horizontal move, then each axis alone (slide
 * along a wall).  A step is only taken when the destination has a support
 * surface inside the step window, so slab openings and room edges stop the
 * walker instead of dropping them onto the storey below.
 */
export function stepWalk(index, pose, dx, dz, options = {}) {
  const opt = optionsWithDefaults(options);
  const attempts = [[dx, dz], [dx, 0], [0, dz]];
  for (const [ax, az] of attempts) {
    if (Math.abs(ax) < EPS && Math.abs(az) < EPS) {
      continue;
    }
    const nx = pose.x + ax;
    const nz = pose.z + az;
    let surface = supportAt(index, nx, nz, pose.feetY, opt);
    if (!surface && opt.gapReach > 0) {
      // Export seams (and the 2 cm lip at the head of a stair) leave cracks
      // narrower than a foot: the body bridges them, so look one stride ahead
      // and keep the destination if the far side is a real surface.
      const length = Math.hypot(ax, az);
      surface = supportAt(index,
        nx + (ax / length) * opt.gapReach,
        nz + (az / length) * opt.gapReach,
        pose.feetY, opt);
    }
    if (!surface) {
      continue;
    }
    if (typeof opt.acceptSupport === 'function' && !opt.acceptSupport(surface.y)) {
      continue;
    }
    // Probe the body at the height it will have on the destination tread: a
    // stair riser ahead is not a wall once the walker is stepping up onto it.
    const probeFeet = Math.max(pose.feetY, surface.y);
    if (blocked(index, pose.x, pose.z, nx, nz, probeFeet, opt)) {
      continue;
    }
    return { x: nx, z: nz, feetY: surface.y, support: surface, moved: true };
  }
  return { x: pose.x, z: pose.z, feetY: pose.feetY, support: null, moved: false };
}
