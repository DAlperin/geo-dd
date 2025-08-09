use differential_dataflow::input::InputSession;
use differential_dataflow::operators::Count;
use differential_dataflow::operators::JoinCore;
use differential_dataflow::operators::Reduce;
use differential_dataflow::operators::Threshold;
use differential_dataflow::operators::arrange::arrangement::ArrangeByKey;
use serde::{Deserialize, Serialize};
use std::cmp::max;

const MAX_LEVEL: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct Rect {
    min_x: i64,
    min_y: i64,
    max_x: i64,
    max_y: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct CellKey {
    level: u8,
    x: i64,
    y: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq, PartialOrd, Ord)]
struct KnnQuery {
    query_id: u64,
    point: (i64, i64),
    k: usize,
    search_radius: i64,
}

// Distance calculation result for KNN
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq, PartialOrd, Ord)]
struct DistanceResult {
    query_id: u64,
    object_id: u64,
    distance_squared: i64,
}

impl Rect {
    fn intersects(&self, other: &Rect) -> bool {
        !(self.max_x <= other.min_x
            || self.min_x >= other.max_x
            || self.max_y <= other.min_y
            || self.min_y >= other.max_y)
    }

    fn contains(&self, other: &Rect) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }
    fn width(&self) -> i64 {
        max(0, self.max_x - self.min_x)
    }
    fn height(&self) -> i64 {
        max(0, self.max_y - self.min_y)
    }
}

// Ok heres the idea.
// This is like a weird quadtree/rtree hybrid. Designed for differential dataflow.
// The idea is that we tile space into dyadic squares, and then cover a rectangle with a minimal set of these squares.
// Levels are dyadic: level 0 is the largest cells; MAX_LEVEL is the smallest.
// We cover a rectangle with a minimal set of dyadic squares by splitting only where
// the rectangle crosses cell boundaries (quadtree-style refinement).

fn cell_size(level: u8) -> i64 {
    // Smallest cells (MAX_LEVEL) have size 1; sizes double as level decreases.
    1 << (MAX_LEVEL - level)
}

fn cell_rect(level: u8, x: i64, y: i64) -> Rect {
    let s = cell_size(level);
    Rect {
        min_x: x * s,
        min_y: y * s,
        max_x: (x + 1) * s,
        max_y: (y + 1) * s,
    }
}

// Find the smallest dyadic **square** size (power of 2) that can contain the rect’s width/height,
// capped so we can still represent it within our [0..=MAX_LEVEL] level scale where size(level)=2^(MAX_LEVEL-level).
fn root_level_for(rect: &Rect) -> u8 {
    let mut need = rect.width().max(rect.height());
    if need <= 0 {
        need = 1; // degenerate -> smallest cell
    }

    // s = next power of two >= need
    let mut s: i64 = 1;
    while s < need {
        s <<= 1;
    }

    // Now s = 2^k. We need level L where 2^(MAX_LEVEL - L) = s.
    // => L = MAX_LEVEL - k, where k is such that (1 << k) == s.
    let mut k: u8 = 0;
    let mut t = s;
    while t > 1 {
        t >>= 1;
        k += 1;
    }

    // Clamp so we don't go negative if s would exceed the level-0 size.
    // (If s > 2^MAX_LEVEL then MAX_LEVEL - k would underflow; clamp to 0.)
    if k > MAX_LEVEL { 0 } else { MAX_LEVEL - k }
}

fn dyadic_cover(rect: Rect) -> Vec<CellKey> {
    if rect.width() == 0 || rect.height() == 0 {
        return Vec::new();
    }

    // Depth-first refinement: emit if fully inside or at MAX_LEVEL; otherwise split.
    fn dfs(level: u8, x: i64, y: i64, target: &Rect, out: &mut Vec<CellKey>) {
        let cr = cell_rect(level, x, y);
        if !cr.intersects(target) {
            return;
        }
        if target.contains(&cr) || level == MAX_LEVEL {
            out.push(CellKey { level, x, y });
            return;
        }
        let child = level + 1;
        let bx = x << 1;
        let by = y << 1;
        dfs(child, bx, by, target, out);
        dfs(child, bx + 1, by, target, out);
        dfs(child, bx, by + 1, target, out);
        dfs(child, bx + 1, by + 1, target, out);
    }

    let mut out = Vec::new();

    // Try to use a single root if possible (smaller than or equal to one level-0 tile).
    let root_level = root_level_for(&rect);
    if root_level > 0 {
        let s = cell_size(root_level);
        let min_x_cell = rect.min_x.div_euclid(s);
        let min_y_cell = rect.min_y.div_euclid(s);
        let max_x_cell = (rect.max_x - 1).div_euclid(s);
        let max_y_cell = (rect.max_y - 1).div_euclid(s);

        // Check if the rectangle spans multiple root-level cells
        if min_x_cell == max_x_cell && min_y_cell == max_y_cell {
            // Rectangle fits in a single root cell
            dfs(root_level, min_x_cell, min_y_cell, &rect, &mut out);
        } else {
            // Rectangle spans multiple cells at root level, enumerate all of them
            for x in min_x_cell..=max_x_cell {
                for y in min_y_cell..=max_y_cell {
                    dfs(root_level, x, y, &rect, &mut out);
                }
            }
        }
    } else {
        // Otherwise, span ALL level-0 tiles that intersect the rect.
        let s0 = cell_size(0);
        // Half-open rect: [min, max). Convert to intersecting tile index range.
        let x0 = rect.min_x.div_euclid(s0);
        let x1 = (rect.max_x - 1).div_euclid(s0) + 1;
        let y0 = rect.min_y.div_euclid(s0);
        let y1 = (rect.max_y - 1).div_euclid(s0) + 1;
        for x in x0..x1 {
            for y in y0..y1 {
                dfs(0, x, y, &rect, &mut out);
            }
        }
    }

    out.sort();
    out
}

// Calculate squared distance from point to rectangle (0 if point is inside)
fn point_to_rect_distance_squared(point: (i64, i64), rect: &Rect) -> i64 {
    let dx = if point.0 < rect.min_x {
        rect.min_x - point.0
    } else if point.0 > rect.max_x {
        point.0 - rect.max_x
    } else {
        0
    };

    let dy = if point.1 < rect.min_y {
        rect.min_y - point.1
    } else if point.1 > rect.max_y {
        point.1 - rect.max_y
    } else {
        0
    };

    dx * dx + dy * dy
}

// Generate all ancestor cells for a given cell (including the cell itself)
// This allows queries to match with objects at any level in the hierarchy
fn cell_ancestors(cell: CellKey) -> Vec<CellKey> {
    let mut ancestors = Vec::new();
    let mut level = cell.level;
    let mut x = cell.x;
    let mut y = cell.y;

    // Add the cell itself
    ancestors.push(cell);

    // Add all ancestors up to level 0
    while level > 0 {
        level -= 1;
        x >>= 1; // parent x coordinate
        y >>= 1; // parent y coordinate
        ancestors.push(CellKey { level, x, y });
    }

    ancestors
}

fn cell_distance_bounds(query_point: (i64, i64), cell: CellKey) -> (i64, i64) {
    let cell_rect = cell_rect(cell.level, cell.x, cell.y);

    // Minimum distance (0 if point is inside cell)
    let min_dist_sq = point_to_rect_distance_squared(query_point, &cell_rect);

    // Maximum distance (to farthest corner)
    let corners = [
        (cell_rect.min_x, cell_rect.min_y),
        (cell_rect.min_x, cell_rect.max_y),
        (cell_rect.max_x, cell_rect.min_y),
        (cell_rect.max_x, cell_rect.max_y),
    ];

    let max_dist_sq = corners
        .iter()
        .map(|&corner| {
            let dx = query_point.0 - corner.0;
            let dy = query_point.1 - corner.1;
            dx * dx + dy * dy
        })
        .max()
        .unwrap();

    (min_dist_sq, max_dist_sq)
}

type ObjectId = u64;
type QueryId = u64;

fn main() {
    timely::execute_from_args(std::env::args(), move |worker| {
        let mut geo_input = InputSession::<i64, _, _>::new();
        let mut query_input = InputSession::<i64, _, _>::new();
        let mut knn_query_input = InputSession::<i64, (u64, (i64, i64), usize), isize>::new();

        let _ = worker.dataflow(|scope| {
            let input_collection = geo_input.to_collection(scope);
            let query_collection = query_input.to_collection(scope);
            let knn_query_collection = knn_query_input.to_collection(scope);

            let covered = input_collection.flat_map(|(object_id, rect)| {
                let cells = dyadic_cover(rect);
                cells.into_iter().map(move |cell| (cell, (object_id, rect)))
            });

            covered.map(|_| ()).count().inspect(|count| {
                println!("Covered collection size: {:?} records", count.0);
            });

            let by_cell = covered.arrange_by_key();

            let query_cells = query_collection.flat_map(|(query_id, rect)| {
                dyadic_cover(rect)
                    .into_iter()
                    .flat_map(|cell| cell_ancestors(cell)) // Expand each query cell to include all ancestors
                    .map(move |cell| (cell, (query_id, rect)))
            });

            query_cells.map(|_| ()).count().inspect(|count| {
                println!("Query cells size: {:?} records", count.0);
            });

            let candidates = query_cells.join_core(
                &by_cell,
                |_k, (query_id, query_rect), (object_id, object_rect)| {
                    Some((*query_id, *query_rect, *object_id, *object_rect))
                },
            );

            candidates.map(|_| ()).count().inspect(|count| {
                println!("Candidate cells size: {:?} records", count.0);
            });

            let hits = candidates
                .filter(|(_query_id, query_rect, _object_id, object_rect)| {
                    query_rect.intersects(object_rect)
                })
                .map(|(query_id, _query_rect, object_id, _object_rect)| (query_id, object_id));

            hits.inspect(|hit| {
                println!("Hit: query_id={}, object_id={}", hit.0.0, hit.0.1);
            });

            let cell_distances = knn_query_collection
                .map(|(query_id, point, k)| ((), (query_id, point, k)))
                .join_core(
                    &covered
                        .map(|(cell, _objects)| ((), cell))
                        .distinct()
                        .arrange_by_key(),
                    |_key, (query_id, point, k), cell| {
                        let (min_dist_sq, max_dist_sq) = cell_distance_bounds(*point, *cell);
                        Some((*query_id, (*cell, min_dist_sq, max_dist_sq, *k)))
                    },
                );

            cell_distances.map(|_| ()).count().inspect(|count| {
                println!("Cell distances size: {:?} records", count.0);
            });

            // base threshold is k-th smallest max_distance (theoretical minimum)
            // plus a safety margin of largest cell spread to account for objects not at cell boundaries
            let closest_cells = cell_distances.reduce(|_query_id, input, output| {
                if let Some((_, _, _, k)) = input.first().map(|((_, _, _, k), _)| ((), (), (), *k))
                {
                    let cell_bounds: Vec<_> = input
                        .iter()
                        .map(|((cell, min_dist_sq, max_dist_sq, _k), diff)| {
                            (*cell, *min_dist_sq, *max_dist_sq, *diff)
                        })
                        .collect();

                    // Find k-th smallest max_distance as base threshold
                    let mut max_distances: Vec<_> = cell_bounds
                        .iter()
                        .map(|(_, _, max_dist, _)| *max_dist)
                        .collect();
                    max_distances.sort();

                    let base_threshold = if max_distances.len() >= k {
                        max_distances[k - 1] // k-th smallest max_distance
                    } else {
                        i64::MAX
                    };

                    // largest difference between max_dist and min_dist in any cell
                    // this accounts for the fact that objects might not be at cell boundaries
                    let max_cell_spread = cell_bounds
                        .iter()
                        .map(|(_, min_dist, max_dist, _)| max_dist - min_dist)
                        .max()
                        .unwrap_or(0);

                    let threshold = base_threshold.saturating_add(max_cell_spread);

                    // Include all cells whose min_dist <= threshold
                    for (cell, min_dist, _max_dist, diff) in cell_bounds {
                        if min_dist <= threshold {
                            output.push((cell, diff));
                        }
                    }
                }
            });

            closest_cells.map(|_| ()).count().inspect(|count| {
                println!("Closest cells size: {:?} records", count.0);
            });

            // Only calculate actual object distances for objects in the closest cells
            let distances = knn_query_collection
                .map(|(query_id, point, k)| (query_id, (point, k)))
                .join_core(
                    &closest_cells
                        .map(|(query_id, cell)| (query_id, cell))
                        .arrange_by_key(),
                    |query_id, (point, k), cell| Some((*query_id, (*cell, *point, *k))),
                )
                .map(|(query_id, (cell, point, k))| (cell, (query_id, point, k)))
                .join_core(
                    &by_cell,
                    |_cell, (query_id, point, k), (object_id, object_rect)| {
                        let actual_dist_sq = point_to_rect_distance_squared(*point, object_rect);
                        Some((*query_id, (*object_id, actual_dist_sq, *k)))
                    },
                );

            distances.map(|_| ()).count().inspect(|count| {
                println!("Distances size: {:?} records", count.0);
            });

            // Group by query and select top-k
            let knn_results_first_pass = distances
                .reduce(|_query_id, input, output| {
                    if let Some((_, _, k)) = input.first().map(|((_, _, k), _)| ((), (), *k)) {
                        let mut candidates: Vec<_> = input
                            .iter()
                            .map(|((obj_id, dist_sq, _k), diff)| (*obj_id, *dist_sq, *diff))
                            .collect();

                        candidates.sort_by_key(|(_, dist, _)| *dist);
                        candidates.truncate(k);

                        for (obj_id, dist_sq, diff) in candidates {
                            output.push(((obj_id, dist_sq), diff));
                        }
                    }
                })
                .map(|(query_id, (obj_id, dist_sq))| (query_id, obj_id, dist_sq));

            knn_results_first_pass.inspect(|((query_id, obj_id, dist_sq), _time, _diff)| {
                println!(
                    "KNN Result: query_id={:?}, object_id={:?}, distance_sq={:?}",
                    query_id, obj_id, dist_sq
                );
            });
        });

        let objects: Vec<(ObjectId, Rect)> = vec![
            (
                1,
                Rect {
                    min_x: 0,
                    min_y: 0,
                    max_x: 2,
                    max_y: 2,
                },
            ),
            (
                2,
                Rect {
                    min_x: 10,
                    min_y: 10,
                    max_x: 12,
                    max_y: 12,
                },
            ),
            (
                3,
                Rect {
                    min_x: 5,
                    min_y: 0,
                    max_x: 7,
                    max_y: 2,
                },
            ),
            (
                4,
                Rect {
                    min_x: 20,
                    min_y: 20,
                    max_x: 22,
                    max_y: 22,
                },
            ),
            (
                5,
                Rect {
                    min_x: 0,
                    min_y: 10,
                    max_x: 2,
                    max_y: 12,
                },
            ),
        ];

        for (object_id, rect) in objects.clone() {
            geo_input.insert((object_id, rect));
        }

        let queries: Vec<(QueryId, Rect)> = vec![
            (
                1,
                Rect {
                    min_x: 1,
                    min_y: 1,
                    max_x: 3,
                    max_y: 3,
                },
            ),
            (
                2,
                Rect {
                    min_x: 9,
                    min_y: 9,
                    max_x: 13,
                    max_y: 13,
                },
            ),
            (
                3,
                Rect {
                    min_x: 4,
                    min_y: -1,
                    max_x: 8,
                    max_y: 3,
                },
            ),
            (
                4,
                Rect {
                    min_x: 25,
                    min_y: 25,
                    max_x: 30,
                    max_y: 30,
                },
            ),
        ];

        for (query_id, rect) in queries {
            query_input.insert((query_id, rect));
        }

        let knn_queries: Vec<(u64, (i64, i64), usize)> =
            vec![(1, (3, 3), 2), (2, (15, 15), 3), (3, (8, 5), 2)];

        for (query_id, point, k) in knn_queries {
            knn_query_input.insert((query_id, point, k));
        }

        geo_input.advance_to(1);
        query_input.advance_to(1);
        knn_query_input.advance_to(1);

        worker.step();

        println!("\n=== Range Query Results ===");
    })
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn cover_aligned_unit_square() {
        // Pick some middle level L (not 0 or MAX) so it’s valid for all MAX_LEVEL >= 2.
        let target_level = MAX_LEVEL / 2;
        let s = cell_size(target_level);
        let x = 2;
        let y = 1;
        let r = Rect {
            min_x: x * s,
            min_y: y * s,
            max_x: (x + 1) * s,
            max_y: (y + 1) * s,
        };
        let cells = dyadic_cover(r);
        assert_eq!(
            cells,
            vec![CellKey {
                level: target_level,
                x,
                y
            }]
        );
    }

    #[test]
    fn cover_misaligned_small() {
        // A small rectangle that will force splitting but doesn’t align to any grid perfectly.
        let r = Rect {
            min_x: 1,
            min_y: 1,
            max_x: 4,
            max_y: 3,
        };
        let cells = dyadic_cover(r);
        assert!(!cells.is_empty());
        for c in &cells {
            assert!(cell_rect(c.level, c.x, c.y).intersects(&r));
        }
    }

    #[test]
    fn cover_spans_levels_and_no_ancestor_conflicts() {
        // Construct a rect that will definitely span levels for any MAX_LEVEL >= 2:
        // We'll take half of a level-(MAX_LEVEL-2) cell and extend a bit into its neighbor.
        let coarse_level = MAX_LEVEL.saturating_sub(2);
        let s_coarse = cell_size(coarse_level);

        // Left half fully inside, right half partial → mixed levels.
        let min_x = s_coarse;
        let min_y = 0;
        let max_x = s_coarse + (s_coarse * 3 / 4); // partial overlap into neighbor
        let max_y = s_coarse;
        let r = Rect {
            min_x,
            min_y,
            max_x,
            max_y,
        };

        let cells = dyadic_cover(r);
        assert!(!cells.is_empty(), "cover should not be empty");

        // 1) It should span multiple levels.
        let levels: HashSet<u8> = cells.iter().map(|c| c.level).collect();
        assert!(
            levels.len() >= 2,
            "expected cover to span multiple levels, got levels: {:?}",
            levels
        );

        // 2) No cell should coexist with its ancestor.
        let set: HashSet<(u8, i64, i64)> = cells.iter().map(|c| (c.level, c.x, c.y)).collect();
        for c in &cells {
            let mut lvl = c.level;
            let mut x = c.x;
            let mut y = c.y;
            while lvl > 0 {
                let parent = (lvl - 1, x >> 1, y >> 1);
                assert!(
                    !set.contains(&parent),
                    "found cell {:?} and its ancestor {:?} both in cover",
                    c,
                    parent
                );
                x >>= 1;
                y >>= 1;
                lvl -= 1;
            }
        }

        // 3) We shouldn't end with all four siblings at the same level.
        let mut by_parent: HashMap<(u8, i64, i64), usize> = HashMap::new();
        for c in &cells {
            if c.level == 0 {
                continue;
            }
            let p = (c.level - 1, c.x >> 1, c.y >> 1);
            *by_parent.entry(p).or_default() += 1;
        }
        for (p, cnt) in by_parent {
            assert_ne!(
                cnt, 4,
                "found four siblings with parent {:?}; parent should have been kept",
                p
            );
        }

        // 4) Sanity: every emitted cell must intersect the rect.
        for c in &cells {
            assert!(
                cell_rect(c.level, c.x, c.y).intersects(&r),
                "cell {:?} does not intersect target rect",
                c
            );
        }
    }

    #[test]
    fn large_rect_crosses_multiple_level0_roots() {
        let s0 = cell_size(0);
        // Span 2.5 level-0 tiles horizontally, 1 tile vertically
        let r = Rect {
            min_x: 0,
            min_y: 0,
            max_x: 2 * s0 + (s0 / 2),
            max_y: s0,
        };

        let cells = dyadic_cover(r);
        assert!(!cells.is_empty(), "cover should not be empty");

        // Compute the level-0 ancestor (root tile) of every emitted cell.
        let mut root_tiles: HashSet<(i64, i64)> = HashSet::new();
        for c in &cells {
            let mut x = c.x;
            let mut y = c.y;
            let mut lvl = c.level;
            while lvl > 0 {
                x >>= 1;
                y >>= 1;
                lvl -= 1;
            }
            root_tiles.insert((x, y));
        }
        assert!(
            root_tiles.len() >= 2,
            "expected to span multiple level-0 tiles, got {root_tiles:?}"
        );

        // No cell should coexist with its ancestor.
        let set: HashSet<(u8, i64, i64)> = cells.iter().map(|c| (c.level, c.x, c.y)).collect();
        for c in &cells {
            let mut x = c.x;
            let mut y = c.y;
            let mut lvl = c.level;
            while lvl > 0 {
                let parent = (lvl - 1, x >> 1, y >> 1);
                assert!(
                    !set.contains(&parent),
                    "found cell {:?} and its ancestor {:?} both in cover",
                    c,
                    parent
                );
                x >>= 1;
                y >>= 1;
                lvl -= 1;
            }
        }

        // No parent should have all four children present (we'd have kept the parent).
        let mut by_parent: HashMap<(u8, i64, i64), usize> = HashMap::new();
        for c in &cells {
            if c.level == 0 {
                continue;
            }
            let p = (c.level - 1, c.x >> 1, c.y >> 1);
            *by_parent.entry(p).or_default() += 1;
        }
        for (p, cnt) in by_parent {
            assert_ne!(
                cnt, 4,
                "found four siblings with parent {:?}; parent should have been kept",
                p
            );
        }

        println!(
            "Cover for rect {:?} spans {} cells: {:?}",
            r,
            cells.len(),
            cells
        );

        // Every emitted cell must intersect the rect.
        for c in &cells {
            assert!(
                cell_rect(c.level, c.x, c.y).intersects(&r),
                "cell {:?} does not intersect target rect",
                c
            );
        }
    }
}
