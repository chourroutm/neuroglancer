/**
 * @license
 * Copyright 2024 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Tests for the measurement support code in `#src/ui/annotations.js`.
 *
 * This is a `browser_test` rather than a `spec` because `#src/ui/annotations.js`
 * transitively imports `#src/webgl/shader.js`, which evaluates
 * `WebGL2RenderingContext.VERTEX_SHADER` at module scope. Importing it under the
 * node project therefore fails at collection with
 * `ReferenceError: WebGL2RenderingContext is not defined`.
 */

import { describe, it, expect } from "vitest";
import type { AnnotationLayerState } from "#src/annotation/annotation_layer_state.js";
import type { CoordinateSpace } from "#src/coordinate_transform.js";
import {
  emptyValidCoordinateSpace,
  makeCoordinateSpace,
} from "#src/coordinate_transform.js";
import type {
  RenderLayerTransform,
  RenderLayerTransformOrError,
} from "#src/render_coordinate_transform.js";
import { WatchableValue } from "#src/trackable_value.js";
import type { UserLayerWithAnnotations } from "#src/ui/annotations.js";
import { getAnnotationPhysicalScales } from "#src/ui/annotations.js";
import * as matrix from "#src/util/matrix.js";

/** Builds a real coordinate space from parallel name/unit/scale lists. */
const space = (
  names: readonly string[],
  units: readonly string[],
  scales: readonly number[],
): CoordinateSpace =>
  makeCoordinateSpace({ names, units, scales: Float64Array.from(scales) });

/**
 * Builds a render-layer transform. Only `rank` and the two dimension mappings
 * are read by `getAnnotationPhysicalScales`; the remaining fields carry the
 * values an identity transform of that rank would have.
 */
const renderLayerTransform = (options: {
  rank: number;
  globalToRenderLayerDimensions?: readonly number[];
  localToRenderLayerDimensions?: readonly number[];
}): RenderLayerTransform => {
  const {
    rank,
    globalToRenderLayerDimensions = [],
    localToRenderLayerDimensions = [],
  } = options;
  return {
    rank,
    unpaddedRank: rank,
    localToRenderLayerDimensions,
    globalToRenderLayerDimensions,
    channelToRenderLayerDimensions: [],
    channelToModelDimensions: [],
    channelSpaceShape: new Uint32Array(0),
    modelToRenderLayerTransform: matrix.createIdentity(Float32Array, rank + 1),
    modelDimensionNames: [],
    layerDimensionNames: [],
  };
};

/**
 * `getAnnotationPhysicalScales` reads exactly three watchable values: the
 * annotation layer's transform, the root (global) coordinate space, and the
 * layer's local coordinate space. A real `AnnotationLayerState` additionally
 * needs an `AnnotationDisplayState` and a `LayerDataSource`, and a real
 * `UserLayerWithAnnotations` needs a `TopLevelLayerListSpecification` (and so a
 * live display/WebGL context), so the two containers are narrow stubs holding
 * real `WatchableValue`s over real `CoordinateSpace`s.
 */
const annotationLayerWithTransform = (
  transform: RenderLayerTransformOrError,
): AnnotationLayerState =>
  ({
    transform: new WatchableValue(transform),
  }) as unknown as AnnotationLayerState;

const layerWithCoordinateSpaces = (
  globalCoordinateSpace: CoordinateSpace,
  localCoordinateSpace: CoordinateSpace = emptyValidCoordinateSpace,
): UserLayerWithAnnotations =>
  ({
    manager: {
      root: { coordinateSpace: new WatchableValue(globalCoordinateSpace) },
    },
    localCoordinateSpace: new WatchableValue(localCoordinateSpace),
  }) as unknown as UserLayerWithAnnotations;

// Markers refer to the clauses of the function's doc comment:
//   C1: returns undefined if the layer transform is not resolved
//   C2: one entry per annotation (render-layer) dimension
//   C3: a dimension whose unit is not length (meters) contributes 0
// and to the mapping/guard behaviour of the implementation:
//   G1: global dimensions map through `globalToRenderLayerDimensions`
//   G2: local dimensions map through `localToRenderLayerDimensions`
//   G3: a render dimension of -1, or at/beyond the rank, is skipped
describe("getAnnotationPhysicalScales", () => {
  // C1: transform not resolved -> undefined
  it("returns undefined when the layer transform is not resolved", () => {
    expect(
      getAnnotationPhysicalScales(
        annotationLayerWithTransform({ error: "Rank mismatch" }),
        layerWithCoordinateSpaces(space(["x", "y"], ["m", "m"], [1e-9, 1e-9])),
      ),
    ).toBeUndefined();
  });

  // C2: one entry per render-layer dimension
  it("returns a Float64Array of length transform.rank", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        renderLayerTransform({
          rank: 3,
          globalToRenderLayerDimensions: [0, 1, 2],
        }),
      ),
      layerWithCoordinateSpaces(
        space(["x", "y", "z"], ["m", "m", "m"], [1e-9, 1e-9, 1e-9]),
      ),
    );
    expect(scaleNm).toBeInstanceOf(Float64Array);
    expect(scaleNm).toHaveLength(3);
  });

  // C2: rank may exceed the number of mapped dimensions (padding dimensions);
  // the extra entries stay 0 rather than being dropped.
  it("keeps an entry for a render dimension no coordinate space maps to", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        renderLayerTransform({
          rank: 4,
          globalToRenderLayerDimensions: [0, 1, 2],
        }),
      ),
      layerWithCoordinateSpaces(
        space(["x", "y", "z"], ["m", "m", "m"], [4e-9, 4e-9, 4e-9]),
      ),
    )!;
    expect(scaleNm).toHaveLength(4);
    expect(scaleNm[3]).toBe(0);
  });

  // Meters -> nanometers, i.e. `scales[d] * 1e9`.
  it("converts meter scales to nanometers", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        renderLayerTransform({
          rank: 2,
          globalToRenderLayerDimensions: [0, 1],
        }),
      ),
      // 4 nm/unit and 40 nm/unit, expressed in meters.
      layerWithCoordinateSpaces(space(["x", "y"], ["m", "m"], [4e-9, 40e-9])),
    )!;
    expect(scaleNm[0]).toBeCloseTo(4, 6);
    expect(scaleNm[1]).toBeCloseTo(40, 6);
  });

  // G1: the global mapping is applied, not assumed to be the identity.
  it("maps global dimensions through globalToRenderLayerDimensions", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        // Global dimensions (x, y, z) become render dimensions (2, 0, 1).
        renderLayerTransform({
          rank: 3,
          globalToRenderLayerDimensions: [2, 0, 1],
        }),
      ),
      layerWithCoordinateSpaces(
        space(["x", "y", "z"], ["m", "m", "m"], [1e-9, 2e-9, 3e-9]),
      ),
    )!;
    expect(scaleNm[0]).toBeCloseTo(2, 6); // global y
    expect(scaleNm[1]).toBeCloseTo(3, 6); // global z
    expect(scaleNm[2]).toBeCloseTo(1, 6); // global x
  });

  // G2: local dimensions are picked up from the layer's local coordinate space.
  it("maps local dimensions through localToRenderLayerDimensions", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        renderLayerTransform({
          rank: 3,
          globalToRenderLayerDimensions: [0, 1],
          localToRenderLayerDimensions: [2],
        }),
      ),
      layerWithCoordinateSpaces(
        space(["x", "y"], ["m", "m"], [4e-9, 4e-9]),
        space(["z"], ["m"], [7e-9]),
      ),
    )!;
    expect(scaleNm[0]).toBeCloseTo(4, 6);
    expect(scaleNm[1]).toBeCloseTo(4, 6);
    expect(scaleNm[2]).toBeCloseTo(7, 6);
  });

  // C3: a non-length unit yields 0. Callers treat a 0 entry as "not a length
  // axis" and skip it, so this is what keeps a time or channel axis out of a
  // measured length, projected axis length, or bounding-box volume.
  it("gives a dimension whose unit is not meters a scale of 0", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        renderLayerTransform({
          rank: 4,
          globalToRenderLayerDimensions: [0, 1, 2, 3],
        }),
      ),
      layerWithCoordinateSpaces(
        // "t" is seconds and "c" is dimensionless; neither is a length.
        space(["x", "y", "t", "c"], ["m", "m", "s", ""], [4e-9, 4e-9, 1, 1]),
      ),
    )!;
    expect(scaleNm[0]).toBeCloseTo(4, 6);
    expect(scaleNm[1]).toBeCloseTo(4, 6);
    expect(scaleNm[2]).toBe(0);
    expect(scaleNm[3]).toBe(0);
  });

  // G3: a render dimension outside `[0, rank)` is skipped. Both of the next two
  // cases pin the observable contract -- no throw, correct length, in-range
  // entries untouched -- rather than the bounds check itself: an out-of-bounds
  // or negative index write to a Float64Array is silently ignored, so the check
  // is defensive. They do guard against a plausible wrong fix such as clamping
  // the index into range.
  it("skips a dimension mapped to render dimension -1", () => {
    const scaleNm = getAnnotationPhysicalScales(
      annotationLayerWithTransform(
        renderLayerTransform({
          rank: 2,
          globalToRenderLayerDimensions: [0, 1, -1],
        }),
      ),
      layerWithCoordinateSpaces(
        space(["x", "y", "z"], ["m", "m", "m"], [4e-9, 4e-9, 9e-9]),
      ),
    )!;
    expect(scaleNm).toHaveLength(2);
    expect(scaleNm[0]).toBeCloseTo(4, 6);
    expect(scaleNm[1]).toBeCloseTo(4, 6);
  });

  // G3: see the note above.
  it("skips a render dimension at or beyond the transform rank", () => {
    let scaleNm: Float64Array | undefined;
    expect(() => {
      scaleNm = getAnnotationPhysicalScales(
        annotationLayerWithTransform(
          renderLayerTransform({
            rank: 2,
            globalToRenderLayerDimensions: [0, 1, 5],
          }),
        ),
        layerWithCoordinateSpaces(
          space(["x", "y", "z"], ["m", "m", "m"], [4e-9, 4e-9, 9e-9]),
        ),
      );
    }).not.toThrow();
    expect(scaleNm).toHaveLength(2);
    expect(scaleNm![0]).toBeCloseTo(4, 6);
    expect(scaleNm![1]).toBeCloseTo(4, 6);
  });
});
