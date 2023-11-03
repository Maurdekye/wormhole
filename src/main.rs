use rand::prelude::*;
use rayon::prelude::*;
use std::iter::*;

type Point = (f64, f64);

#[derive(Clone, Debug)]
struct Space {
    bounds: (Point, Point),
    holes: Vec<(Point, Point)>,
    hole_multiplier: f64
}

fn sq_dist(a: &Point, b: &Point) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

fn dist(a: &Point, b: &Point) -> f64 { sq_dist(a, b).sqrt() }

struct GridIter {
    start: (usize, usize),
    end: (usize, usize),
    current: (usize, usize)
}

impl GridIter {
    fn between(from: &(usize, usize), to: &(usize, usize)) -> GridIter {
        GridIter {
            start: from.clone(),
            end: to.clone(),
            current: from.clone()
        }
    }
}

impl Iterator for GridIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.1 >= self.end.1 {
            None
        } else {
            let response = self.current;
            self.current.0 += 1;
            if self.current.0 >= self.end.0 {
                self.current.0 = self.start.0;
                self.current.1 += 1;
            }
            Some(response)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let bound = (self.end.0 - self.current.0) + (self.end.1 - self.current.1) * (self.end.0 - self.start.0);
        (bound, Some(bound))
    }
}

impl Space {
    fn new() -> Space {
        Space {
            bounds: ((0.0, 0.0), (1.0, 1.0)),
            holes: Vec::new(),
            hole_multiplier: 0.0
        }
    }

    fn test_with_dist_fn(&self, start: &Point, end: &Point, dist_fn: &dyn Fn(&Point, &Point) -> f64) -> f64 {
        let mut travel_dist = dist_fn(start, end);
        for (hole_a, hole_b) in self.holes.iter()  {
            let a_dist = dist_fn(start, hole_a);
            let b_dist = dist_fn(start, hole_b);
            let hole_dist = if a_dist < b_dist {
                a_dist + dist_fn(hole_b, end)
            } else {
                b_dist + dist_fn(hole_a, end)
            } + if self.hole_multiplier > f64::default() {
                dist_fn(hole_a, hole_b)
            } else { 
                f64::default() 
            };
            if hole_dist < travel_dist {
                travel_dist = hole_dist;
            }
        }
        travel_dist
    }

    fn test(&self, start: &Point, end: &Point) -> f64 {
        self.test_with_dist_fn(start, end, &dist)
    }

    fn permute(&self, grid_density: usize, random_placement: bool) -> f64 {
        let cell_width = (self.bounds.1.0 - self.bounds.0.0) / grid_density as f64;
        let cell_height = (self.bounds.1.1 - self.bounds.0.1) / grid_density as f64;
        let total: f64 = GridIter::between(&(0, 0), &(grid_density, grid_density)).collect::<Vec<_>>().par_iter().map(|(start_x, start_y)| {
            let mut rng = rand::thread_rng();
            let mut semitotal = 0.0;
            let mut start = (self.bounds.0.0 + cell_width * *start_x as f64, self.bounds.0.1 + cell_height * *start_y as f64);
            if random_placement {
                start.0 += rng.gen::<f64>() * cell_width;
                start.1 += rng.gen::<f64>() * cell_height;
            } else {
                start.0 += cell_width / 2f64;
                start.1 += cell_height / 2f64;
            }
            for (end_x, end_y) in GridIter::between(&(0, 0), &(grid_density, grid_density)) {
                let mut end = (self.bounds.0.0 + cell_width * end_x as f64, self.bounds.0.1 + cell_height * end_y as f64);

                if random_placement {
                    end.0 += rng.gen::<f64>() * cell_width;
                    end.1 += rng.gen::<f64>() * cell_height;
                } else {
                    end.0 += cell_width / 2f64;
                    end.1 += cell_height / 2f64;
                }

                semitotal += self.test(&start, &end);
            }
            semitotal
        }).sum();
        total / grid_density.pow(4) as f64
    }

    fn gradient_descent(&mut self, density: usize, temperature: f64, attenuation: f64) -> Vec<(f64, Point, Point)> {
        let mut otherself = self.clone();
        self.holes.iter_mut().enumerate().map(|(i, hole)| {
            let neutral = otherself.permute(density, true);

            otherself.holes[i].0.0 = hole.0.0 + attenuation;
            let start_x_gradient = otherself.permute(density, true);

            otherself.holes[i].0.0 = hole.0.0;
            otherself.holes[i].0.1 = hole.0.1 + attenuation;
            let start_y_gradient = otherself.permute(density, true);

            otherself.holes[i].0.1 = hole.0.1;
            otherself.holes[i].1.0 = hole.1.0 + attenuation;
            let end_x_gradient = otherself.permute(density, true);

            otherself.holes[i].1.0 = hole.1.0;
            otherself.holes[i].1.1 = hole.1.1 + attenuation;
            let end_y_gradient = otherself.permute(density, true);

            otherself.holes[i].1.1 = hole.1.1;

            let start_gradient = (neutral - start_x_gradient, neutral - start_y_gradient);
            let end_gradient = (neutral - end_x_gradient, neutral - end_y_gradient);

            hole.0.0 += start_gradient.0 * temperature;
            hole.0.1 += start_gradient.1 * temperature;
            hole.1.0 += end_gradient.0 * temperature;
            hole.1.1 += end_gradient.1 * temperature;

            otherself.holes[i] = *hole;

            (neutral, start_gradient, end_gradient)
        }).collect()
    }
}

fn main() {
    let mut space = Space::new();
    space.holes.push(((0.0, 0.0), (1.0, 1.0)));
    for i in 1.. {

        let results = space.gradient_descent(64, 100.0 as f64, 0.1 / i as f64);
        let (score, start_g, end_g) = results.first().unwrap();
        println!("Iteration {}: score: {:.5}, hole: {:?}, start gradient of {:?}, end gradient of {:?}", i, score, space.holes[0], start_g, end_g);

        // let result = space.permute(128, true);
        // avg += result;
        // println!("Iteration {}: Result: {}, Overall: {}", i, result, avg / i as f64);
    }
}
