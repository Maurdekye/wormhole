use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::map::map_pixels;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::{iter::*, fs};

type Position = (f64, f64);

#[derive(Clone, Debug)]
struct Space {
    bounds: (Position, Position),
    holes: Vec<(Position, Position)>,
    hole_multiplier: f64,
}

fn sq_dist(a: &Position, b: &Position) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

fn dist(a: &Position, b: &Position) -> f64 {
    sq_dist(a, b).sqrt()
}

struct GridIter {
    start: (usize, usize),
    end: (usize, usize),
    current: (usize, usize),
}

impl GridIter {
    fn between(from: &(usize, usize), to: &(usize, usize)) -> GridIter {
        GridIter {
            start: from.clone(),
            end: to.clone(),
            current: from.clone(),
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
        let bound = (self.end.0 - self.current.0)
            + (self.end.1 - self.current.1) * (self.end.0 - self.start.0);
        (bound, Some(bound))
    }
}

impl Space {
    fn new() -> Space {
        Space {
            bounds: ((0.0, 0.0), (1.0, 1.0)),
            holes: Vec::new(),
            hole_multiplier: 0.0,
        }
    }

    fn test_with_dist_fn(
        &self,
        start: &Position,
        end: &Position,
        dist_fn: &dyn Fn(&Position, &Position) -> f64,
    ) -> f64 {
        let mut travel_dist = dist_fn(start, end);
        for (hole_a, hole_b) in self.holes.iter() {
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

    fn test(&self, start: &Position, end: &Position) -> f64 {
        self.test_with_dist_fn(start, end, &dist)
    }

    fn permute<T>(&self, grid_density: usize, random_placement: &mut Option<T>) -> f64
    where
        T: Rng,
    {
        let cell_width = (self.bounds.1 .0 - self.bounds.0 .0) / grid_density as f64;
        let cell_height = (self.bounds.1 .1 - self.bounds.0 .1) / grid_density as f64;
        let offsets = GridIter::between(&(0, 0), &(grid_density, grid_density))
            .map(|_| match random_placement {
                None => (cell_width / 2f64, cell_height / 2f64),
                Some(rng) => (
                    rng.gen::<f64>() * cell_width,
                    rng.gen::<f64>() * cell_height,
                ),
            })
            .collect::<Vec<Position>>();
        let total: f64 = GridIter::between(&(0, 0), &(grid_density, grid_density))
            .collect::<Vec<_>>()
            .par_iter()
            .enumerate()
            .map(|(si, (start_x, start_y))| {
                let start = (
                    self.bounds.0 .0 + cell_width * *start_x as f64 + offsets[si].0,
                    self.bounds.0 .1 + cell_height * *start_y as f64 + offsets[si].1,
                );
                GridIter::between(&(0, 0), &(grid_density, grid_density))
                    .enumerate()
                    .map(|(ei, (end_x, end_y))| {
                        let end = (
                            self.bounds.0 .0 + cell_width * end_x as f64 + offsets[ei].0,
                            self.bounds.0 .1 + cell_height * end_y as f64 + offsets[ei].1,
                        );
                        self.test_with_dist_fn(&start, &end, &dist)
                    })
                    .sum::<f64>()
            })
            .sum();
        total / grid_density.pow(4) as f64
    }

    fn gradient_descent(
        &mut self,
        density: usize,
        random_placement: bool,
        temperature: f64,
        epsilon: f64,
    ) -> (f64, Vec<(Position, Position)>) {
        let mut otherself = self.clone();
        let iteration_seed = rand::random::<u64>();
        let get_rng = || {
            if random_placement {
                Some::<ChaCha8Rng>(SeedableRng::seed_from_u64(iteration_seed))
            } else {
                None
            }
        };
        let neutral = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
        let gradients = self
            .holes
            .iter_mut()
            .enumerate()
            .map(|(i, hole)| {
                otherself.holes[i].0 .0 = hole.0 .0 + epsilon;
                let start_x_gradient = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
                otherself.holes[i].0 .0 = hole.0 .0;

                otherself.holes[i].0 .1 = hole.0 .1 + epsilon;
                let start_y_gradient = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
                otherself.holes[i].0 .1 = hole.0 .1;

                otherself.holes[i].1 .0 = hole.1 .0 + epsilon;
                let end_x_gradient = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
                otherself.holes[i].1 .0 = hole.1 .0;

                otherself.holes[i].1 .1 = hole.1 .1 + epsilon;
                let end_y_gradient = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
                otherself.holes[i].1 .1 = hole.1 .1;

                let start_gradient = (
                    (neutral - start_x_gradient) / epsilon,
                    (neutral - start_y_gradient) / epsilon,
                );
                let end_gradient = (
                    (neutral - end_x_gradient) / epsilon,
                    (neutral - end_y_gradient) / epsilon,
                );

                (start_gradient, end_gradient)
            })
            .collect::<Vec<_>>();
        (
            neutral,
            gradients
                .into_iter()
                .zip(self.holes.iter_mut())
                .map(|((start, end), hole)| {
                    hole.0 .0 += start.0 * temperature;
                    hole.0 .1 += start.1 * temperature;
                    hole.1 .0 += end.0 * temperature;
                    hole.1 .1 += end.1 * temperature;
                    (start, end)
                })
                .collect(),
        )
    }

    fn render(&self) -> RgbImage {
        let dims = (1024, 1024);
        let center = ((dims.0 / 2) as f32, (dims.1 / 2) as f32);
        let max_distance = f32::hypot(center.0, center.1);
        let mut img = map_pixels(&RgbImage::new(dims.0, dims.1), |x, y, _| {
            let dist: f32 = f32::hypot((x as i32 - center.0 as i32) as f32, (y as i32 - center.1 as i32) as f32);
            let factor = (max_distance - dist) / max_distance;
            let tan = Rgb([210, 180, 140]);
            Rgb([
                (255.0 * factor + tan[0] as f32 * (1.0 - factor)) as u8,
                (255.0 * factor + tan[1] as f32 * (1.0 - factor)) as u8,
                (255.0 * factor + tan[2] as f32 * (1.0 - factor)) as u8,
            ])
        });
        draw_line_segment_mut(&mut img, (0.0, center.1), (dims.0 as f32, center.1), Rgb([220, 220, 220]));
        draw_line_segment_mut(&mut img, (center.0, 0.0), (center.0, dims.1 as f32), Rgb([220, 220, 220]));
        for hole in self.holes.iter() {
            let (start, end) = hole;
            let (bound_width, bound_height) = (
                self.bounds.1 .0 - self.bounds.0 .0,
                self.bounds.1 .1 - self.bounds.0 .1,
            );
            let start_pixel = (
                (((start.0 - self.bounds.0 .0) / bound_width) * dims.0 as f64) as f32,
                (((start.1 - self.bounds.0 .1) / bound_height) * dims.1 as f64) as f32,
            );
            let end_pixel = (
                (((end.0 - self.bounds.0 .0) / bound_width) * dims.0 as f64) as f32,
                (((end.1 - self.bounds.0 .1) / bound_height) * dims.1 as f64) as f32,
            );
            draw_filled_circle_mut(&mut img, (|(a,b)|(a as i32, b as i32))(start_pixel), 2, Rgb([0, 0, 0]));
            draw_filled_circle_mut(&mut img, (|(a,b)|(a as i32, b as i32))(end_pixel), 2, Rgb([0, 0, 0]));
            draw_line_segment_mut(&mut img, start_pixel, end_pixel, Rgb([128, 128, 128]));
        }
        img
    }
}

fn main() -> std::io::Result<()> {
    let mut space = Space::new();
    space.holes.push(((0.0, 0.0), (1.0, 0.0)));
    fs::create_dir_all("output")?;
    space.holes.push(((0.0, 1.0), (1.0, 1.0)));
    for i in 1.. {
        // let temp = 1.01f64.powi(-i);
        let temp = 1.0;
        let (loss, _) = space.gradient_descent(128, false, temp, 1e-10);
        println!(
            "Iteration {}: temp: {:.3}, loss: {:.10}, holes:",
            i, temp, loss
        );
        for hole in space.holes.iter() {
            println!("{:?}", hole);
        }
        let img = space.render();
        img.save(format!("output/iter-{}.png", i)).unwrap();
    }
    Ok(())
}
