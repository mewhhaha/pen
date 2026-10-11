//! Native compute as a service: the shortest 4-connected path over a grid,
//! by A*. The game sends the grid (`a` columns by `b` rows of cells, 0 open
//! and anything else blocked) and the start and goal cells (`c` and `d`,
//! row * columns + column); the answer's values are the path's cells from
//! start to goal and its value their count, or none when the goal cannot be
//! reached. Ties break the same way every time, so the same grid gives the
//! same path on every machine.
//!
//! It is the example of a native library behind the services boundary: the
//! game hands over plain numbers and gets plain numbers back, and a replay
//! reads the path from the recording instead of searching again.

use pen_protocol::{pathfinding_op, service};

use super::{Answer, Request, Service};

pub struct Pathfinding;

/// A whole number from a request's lane.
fn whole(value: f32) -> Option<usize> {
    (value >= 0.0 && value.fract() == 0.0).then_some(value as usize)
}

pub use pen_runtime::pathfinding::find;

impl Service for Pathfinding {
    fn code(&self) -> u32 {
        service::PATHFINDING
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if request.operation != pathfinding_op::FIND {
            return Some(Answer::unknown());
        }
        let [columns, rows, start, goal] = request.numbers.map(whole);
        let (Some(columns), Some(rows), Some(start), Some(goal)) = (columns, rows, start, goal) else {
            return Some(Answer::failed());
        };
        if columns == 0 || columns.checked_mul(rows) != Some(request.values.len()) {
            return Some(Answer::failed());
        }
        let open: Vec<bool> = request.values.iter().map(|cell| *cell == 0.0).collect();
        Some(match find(columns, &open, start, goal) {
            Some(path) => Answer {
                value: path.len() as f32,
                values: path.into_iter().map(|cell| cell as f32).collect(),
                ..Answer::ok(0.0)
            },
            None => Answer::ok(0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid from rows of `.` (open) and `#` (blocked).
    fn grid(rows: &[&str]) -> (usize, Vec<bool>) {
        (rows[0].len(), rows.iter().flat_map(|row| row.chars().map(|cell| cell == '.')).collect())
    }

    #[test]
    fn finds_the_shortest_way_round_walls() {
        let (columns, open) = grid(&["....", ".##.", "...#", "#..."]);
        let path = find(columns, &open, 0, 15).unwrap();
        // Down the left side and along the bottom: 7 cells, each a step from the last.
        assert_eq!(path.len(), 7);
        assert_eq!((path[0], path[6]), (0, 15));
        assert!(path.windows(2).all(|step| step[0].abs_diff(step[1]) == 1 || step[0].abs_diff(step[1]) == columns));
        assert!(path.iter().all(|cell| open[*cell]));
        // The same path every time.
        assert_eq!(find(columns, &open, 0, 15).unwrap(), path);
        assert_eq!(find(columns, &open, 5, 0), None, "a blocked start");
        let (columns, open) = grid(&[".#.", ".#.", ".#."]);
        assert_eq!(find(columns, &open, 0, 2), None, "a wall between");
        assert_eq!(find(columns, &open, 0, 0), Some(vec![0]));
    }

    #[test]
    fn answers_with_the_cells_or_fails_on_a_bad_grid() {
        let values: Vec<f32> = [0, 0, 1, 0, 0, 0].iter().map(|cell| *cell as f32).collect();
        let ask = |numbers: [f32; 4]| Pathfinding.call(&Request { id: 1, operation: pathfinding_op::FIND, numbers, values: &values }).unwrap();
        let answer = ask([3.0, 2.0, 0.0, 5.0]);
        assert_eq!(answer.value, 4.0);
        assert_eq!(answer.values, [0.0, 1.0, 4.0, 5.0]);
        assert_eq!(ask([4.0, 2.0, 0.0, 5.0]).status, pen_protocol::service_status::FAILED);
        assert_eq!(ask([3.0, 2.0, 0.5, 5.0]).status, pen_protocol::service_status::FAILED);
    }
}
