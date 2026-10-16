use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The shortest path from `start` to `goal` over `open` cells, `columns` to a
/// row, start and goal included; None when there is none.
pub fn find(columns: usize, open: &[bool], start: usize, goal: usize) -> Option<Vec<usize>> {
    if columns == 0 || start >= open.len() || goal >= open.len() || !open[start] || !open[goal] {
        return None;
    }
    let rows = open.len() / columns;
    let place = |cell: usize| (cell % columns, cell / columns);
    let (goal_x, goal_y) = place(goal);
    let estimate = |cell: usize| {
        let (x, y) = place(cell);
        x.abs_diff(goal_x) + y.abs_diff(goal_y)
    };
    let mut cost = vec![usize::MAX; open.len()];
    let mut from = vec![usize::MAX; open.len()];
    // By estimated length, then nearness to the goal, then cell: the same
    // order on every machine.
    let mut frontier = BinaryHeap::new();
    cost[start] = 0;
    frontier.push(Reverse((estimate(start), estimate(start), start)));
    while let Some(Reverse((_, _, cell))) = frontier.pop() {
        if cell == goal {
            let mut path = vec![goal];
            while let Some(&last) = path.last()
                && last != start
            {
                path.push(from[last]);
            }
            path.reverse();
            return Some(path);
        }
        let (x, y) = place(cell);
        let neighbours = [
            (y > 0).then(|| cell - columns),
            (x > 0).then(|| cell - 1),
            (x + 1 < columns).then(|| cell + 1),
            (y + 1 < rows).then(|| cell + columns),
        ];
        for next in neighbours.into_iter().flatten() {
            let through = cost[cell] + 1;
            if open[next] && through < cost[next] {
                cost[next] = through;
                from[next] = cell;
                frontier.push(Reverse((through + estimate(next), estimate(next), next)));
            }
        }
    }
    None
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
}
