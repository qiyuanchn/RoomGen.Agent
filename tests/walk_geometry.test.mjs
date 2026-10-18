// Geometry tests for tools/scene3d_viewer/walk.mjs.
//
// These build tiny GLB-like triangle sets (floor slabs, a stair, a hole, a
// wall, a table) and drive the same walker the browser uses.  Run with
// `node tests/walk_geometry.test.mjs`; tests/test_walk_geometry.py does that
// from the Python suite.

import assert from 'node:assert/strict';
import {
  WALK_DEFAULTS,
  addTriangle,
  blocked,
  createWalkIndex,
  findStandingSpot,
  stepWalk,
  supportAt,
  supportBelow,
} from '../tools/scene3d_viewer/walk.mjs';

const EYE = 1.5;

function addQuad(index, points) {
  addTriangle(index, ...points[0], ...points[1], ...points[2]);
  addTriangle(index, ...points[0], ...points[2], ...points[3]);
}

// A top face with a +Y normal: CCW seen from above.
function addTop(index, x0, x1, z0, z1, y) {
  addQuad(index, [
    [x0, y, z0],
    [x0, y, z1],
    [x1, y, z1],
    [x1, y, z0],
  ]);
}

function addBox(index, x0, x1, z0, z1, y0, y1) {
  addTop(index, x0, x1, z0, z1, y1);
  // side faces (normal.y = 0) -> obstacle grid
  addQuad(index, [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]]);
  addQuad(index, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]);
  addQuad(index, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]);
  addQuad(index, [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]);
}

function stairScene() {
  const index = createWalkIndex();
  addTop(index, -1, 0, 0, 2, 0);
  const steps = 10;
  const rise = 0.25;
  const tread = 0.3;
  for (let step = 0; step < steps; step += 1) {
    addBox(index, step * tread, (step + 1) * tread, 0, 2, 0, (step + 1) * rise);
  }
  addTop(index, steps * tread, steps * tread + 2, 0, 2, steps * rise);
  return { index, top: steps * rise };
}

function testSupportComesFromGeometry() {
  const { index } = stairScene();
  assert.equal(supportAt(index, -0.4, 1.0, 0).y, 0);
  assert.equal(supportAt(index, 0.15, 1.0, 0).y, 0.25);
  assert.equal(supportAt(index, 1.35, 1.0, 1.0).y, 1.25);
  const top = supportAt(index, 3.1, 1.0, 2.5);
  assert.ok(top && Math.abs(top.y - 2.5) < 1e-6);
}

function testWalkerClimbsStairsWithoutAFixedEyeHeight() {
  const { index, top } = stairScene();
  let pose = { x: -0.5, z: 1.0, feetY: 0 };
  const heights = new Set([pose.feetY]);
  for (let step = 0; step < 120; step += 1) {
    const next = stepWalk(index, pose, 0.08, 0);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
    heights.add(Number(pose.feetY.toFixed(3)));
  }
  assert.ok(pose.x > 3.2, `walker should reach the upper floor, stopped at x=${pose.x}`);
  assert.ok(Math.abs(pose.feetY - top) < 1e-6, `feet ${pose.feetY} != ${top}`);
  assert.ok(heights.size >= 10, 'the surface height should follow the treads, not a constant');
  assert.notEqual(pose.feetY + EYE, EYE);
}

function testWallStopsTheWalker() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  addBox(index, 0, 4, 2.0, 2.2, 0, 3.0);
  let pose = { x: 2.0, z: 1.0, feetY: 0 };
  for (let step = 0; step < 40; step += 1) {
    const next = stepWalk(index, pose, 0, 0.1);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.z < 2.0 - WALK_DEFAULTS.bodyRadius + 0.05,
            `walker entered the wall: z=${pose.z}`);
  assert.ok(pose.z > 1.4, 'walker should approach the wall');
}

function testFurnitureStopsTheWalker() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  addBox(index, 1.4, 2.6, 1.6, 2.8, 0, 0.95);  // dining table
  let pose = { x: 2.0, z: 0.8, feetY: 0 };
  for (let step = 0; step < 30; step += 1) {
    const next = stepWalk(index, pose, 0, 0.1);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.z < 1.6 - WALK_DEFAULTS.bodyRadius + 0.05,
            `walker entered the table: z=${pose.z}`);
}

function testSlabOpeningIsNotWalkable() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 3, 0);             // lower storey
  addTop(index, 0, 1.5, 0, 3, 2.5);          // upper slab, left of the hole
  addTop(index, 2.5, 4, 0, 3, 2.5);          // upper slab, right of the hole
  let pose = { x: 1.0, z: 1.5, feetY: 2.5 };
  for (let step = 0; step < 30; step += 1) {
    const next = stepWalk(index, pose, 0.1, 0);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.x < 1.6, `walker crossed the slab opening: x=${pose.x}`);
  assert.ok(Math.abs(pose.feetY - 2.5) < 1e-6, 'walker fell through the opening');
  // The lower floor is really there - the move is refused because the drop is
  // beyond one step, not because the scene has no geometry below.
  const below = supportBelow(index, 2.0, 1.5, 3.0, -1.0);
  assert.ok(below && Math.abs(below.y) < 1e-6);
  assert.equal(supportAt(index, 2.0, 1.5, 2.5), null);
}

function testNarrowSeamIsCrossedButAWideGapIsNot() {
  const seam = createWalkIndex();
  addTop(seam, 0, 1.95, 0, 3, 0);
  addTop(seam, 2.05, 4, 0, 3, 0);   // 10 cm export seam
  let pose = { x: 1.0, z: 1.5, feetY: 0 };
  for (let step = 0; step < 40; step += 1) {
    const next = stepWalk(seam, pose, 0.08, 0);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.x > 2.05, `walker did not cross the 10 cm seam: x=${pose.x}`);

  const gap = createWalkIndex();
  addTop(gap, 0, 1.9, 0, 3, 0);
  addTop(gap, 2.3, 4, 0, 3, 0);     // 40 cm void
  pose = { x: 1.0, z: 1.5, feetY: 0 };
  for (let step = 0; step < 40; step += 1) {
    const next = stepWalk(gap, pose, 0.08, 0);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.x < 1.95, `walker crossed the 40 cm void: x=${pose.x}`);
}

function testTableTopIsNotAClimbableStep() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  addBox(index, 1.0, 2.0, 1.0, 2.0, 0, 1.0);
  const support = supportAt(index, 1.5, 1.5, 0);
  assert.ok(support && Math.abs(support.y) < 1e-6,
            'within the step window the floor wins, the table top is too high');
  assert.ok(blocked(index, 0.6, 1.5, 0.9, 1.5, 0, WALK_DEFAULTS));
}

function testWaistHighPanelStopsTheWalker() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  // A thin floating top (the real dining table: legs at the corners, a 6 cm
  // slab at waist height) must not be walked through between probe heights.
  addBox(index, 1.2, 2.8, 1.5, 2.5, 0.70, 0.76);
  let pose = { x: 2.0, z: 0.8, feetY: 0 };
  for (let step = 0; step < 40; step += 1) {
    const next = stepWalk(index, pose, 0, 0.1);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.z < 1.5 - WALK_DEFAULTS.bodyRadius + 0.05,
            `walker passed through the table top: z=${pose.z}`);
  assert.ok(pose.z > 1.0, 'walker should approach the table top');
}

function testHighShelfDoesNotBlockTheBody() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  addBox(index, 1.2, 2.8, 1.5, 2.5, 1.80, 1.85);
  let pose = { x: 2.0, z: 0.8, feetY: 0 };
  for (let step = 0; step < 40; step += 1) {
    const next = stepWalk(index, pose, 0, 0.1);
    if (!next.moved) {
      break;
    }
    pose = { x: next.x, z: next.z, feetY: next.feetY };
  }
  assert.ok(pose.z > 1.6, `head-height shelf blocked the body: z=${pose.z}`);
}

function testStandingSpotAvoidsFurniture() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  addBox(index, 1.5, 2.5, 1.5, 2.5, 0, 0.95);
  const spot = findStandingSpot(index, 2.0, 2.0, 0);
  assert.ok(spot, 'a free spot should exist next to the furniture');
  const clearance = WALK_DEFAULTS.bodyRadius - 1e-6;
  const outside = spot.x < 1.5 - clearance || spot.x > 2.5 + clearance
    || spot.z < 1.5 - clearance || spot.z > 2.5 + clearance;
  assert.ok(outside, `spot ${spot.x},${spot.z} is still inside the furniture`);
  assert.ok(Math.abs(spot.feetY) < 1e-6, `spot is not on the floor: ${spot.feetY}`);
  assert.ok(!blocked(index, spot.x, spot.z, spot.x + 0.06, spot.z, spot.feetY));
}

function testStandingSpotFindsTheSlabAroundAnOpening() {
  const index = createWalkIndex();
  addTop(index, 0, 1.5, 0, 3, 2.5);   // upper slab, left of the hole
  addTop(index, 2.5, 4, 0, 3, 2.5);   // upper slab, right of the hole
  const spot = findStandingSpot(index, 2.0, 1.5, 2.5);
  assert.ok(spot, 'the slab next to the opening should be found');
  assert.ok(Math.abs(spot.feetY - 2.5) < 1e-6,
            `spot fell to another storey: ${spot.feetY}`);
  assert.ok(spot.x < 1.5 || spot.x > 2.5, `spot stays over the hole: x=${spot.x}`);
}

function testStandingSpotKeepsAnOpenFloorPoint() {
  const index = createWalkIndex();
  addTop(index, 0, 4, 0, 4, 0);
  const spot = findStandingSpot(index, 2.0, 2.0, 0);
  assert.ok(spot);
  assert.equal(spot.x, 2.0);
  assert.equal(spot.z, 2.0);
}

function testSlopeIsSupportAndObstacle() {
  const index = createWalkIndex();
  addQuad(index, [
    [0, 0, 0],
    [0, 0, 1],
    [2, 0.4, 1],
    [2, 0.4, 0],
  ]);
  const support = supportAt(index, 1.0, 0.5, 0.0);
  assert.ok(support && support.y > 0.1 && support.y < 0.3);
}

testSupportComesFromGeometry();
testWalkerClimbsStairsWithoutAFixedEyeHeight();
testWallStopsTheWalker();
testFurnitureStopsTheWalker();
testSlabOpeningIsNotWalkable();
testNarrowSeamIsCrossedButAWideGapIsNot();
testTableTopIsNotAClimbableStep();
testWaistHighPanelStopsTheWalker();
testHighShelfDoesNotBlockTheBody();
testStandingSpotAvoidsFurniture();
testStandingSpotFindsTheSlabAroundAnOpening();
testStandingSpotKeepsAnOpenFloorPoint();
testSlopeIsSupportAndObstacle();
console.log('walk geometry: 13 tests passed');
