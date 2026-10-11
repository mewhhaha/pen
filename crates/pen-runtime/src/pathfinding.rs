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
