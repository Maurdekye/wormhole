use core::panic;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::map::map_pixels;
use rand::distributions::uniform::SampleBorrow;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::f32::consts::{self, PI};
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, iter::*};

type Position = (f64, f64);

fn dist(a: &Position, b: &Position) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.hypot(dy)
    // dx.abs() + dy.abs()
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

trait Train: Clone {
    fn loss(&self, seed: Option<u64>) -> f64;
    fn get_params(&mut self) -> Vec<RefCell<f64>>;
    
    fn render(&self) -> Option<RgbImage> { None }

    fn gradient_descent(
        &mut self,
        temperature: f64,
        epsilon: f64,
        seed: Option<u64>,
    ) -> (f64, u64, Vec<f64>) {
        let mut otherself = self.clone();
        let seed = seed.unwrap_or_else(|| rand::random::<u64>());
        let base_loss = otherself.loss(Some(seed));
        let gradients = self.get_params().iter().enumerate().map(|(i, parameter)| {
            *otherself.get_params()[i].borrow_mut() = *parameter.borrow() + epsilon;
            let gradient_loss = otherself.loss(Some(seed));
            *otherself.get_params()[i].borrow_mut() = *parameter.borrow();
            if base_loss == gradient_loss {
                0.0
            } else {
                (base_loss - gradient_loss) / epsilon
            }
        }).collect::<Vec<_>>();
        self.get_params().iter_mut().enumerate().for_each(|(i, parameter)| {
            *parameter.borrow_mut() += gradients[i] * temperature;
        });
        (base_loss, seed, gradients)
    }

    fn train_and_save(
        &mut self,
        name: String,
        iters: usize,
        learn_rate: f64,
        loss_window_size: usize,
        save_video: Option<FfmpegOptions>,
    ) -> std::io::Result<()> {
        println!("Beginning training of {}", name);
        fs::create_dir_all(format!("output/{}/", name))?;
        let mut logfile = File::create(format!("output/{}.log", name))?;
        let mut timestamp = SystemTime::now();
        let mut loss_eval_window = VecDeque::with_capacity(loss_window_size);
        let mut has_render = true;
        for i in 0..iters {
            let (loss, seed, gradients) = if i == 0 {
                // default values to record the 0th iteration
                let seed = rand::random::<u64>();
                (
                    self.loss(None),
                    seed,
                    self.get_params().iter().map(|_| 0.0).collect::<Vec<_>>()
                )
            } else {
                let (loss, seed, gradients) = self.gradient_descent(learn_rate, 1e-10, None);
                
                loss_eval_window.push_back(loss.log10());
                if loss_eval_window.len() > loss_window_size {
                    loss_eval_window.pop_front();
                }

                let new_time = SystemTime::now();
                let delta = new_time.duration_since(timestamp).unwrap();
                timestamp = new_time;

                let gradient_magnitudes = gradients.iter().map(|g| g.log10()).collect::<Vec<_>>();
                
                println!(
                    "Iteration {}: took: {:.4}s, loss: {:.10}, gradients: {}",
                    i,
                    delta.as_secs_f64(),
                    loss,
                    gradient_magnitudes
                        .iter()
                        .map(|f| format!("{:.2}", f))
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                (loss, seed, gradients)
            };

            let loss_window_slope = slope(&loss_eval_window);

            writeln!(
                logfile,
                "{},{},{},{},{},{:?},{:?}",
                i,
                seed,
                timestamp.duration_since(UNIX_EPOCH).unwrap().as_millis(),
                loss,
                loss_window_slope,
                self.get_params(),
                gradients
            )?;

            if loss_eval_window.len() >= loss_window_size && loss_window_slope < 0.01
            {
                println!("Gradients settled, ending training");
                break;
            }

            match self.render() {
                Some(img) => img.save(format!("output/{}/iter-{}.png", name, i)).unwrap(),
                None => {
                    has_render = false;
                }
            };
        }

        match (has_render, save_video)  {
            (true, Some(ffmpeg_options)) => {
                Command::new("ffmpeg")
                    .args([
                        "-r",
                        format!("{}", ffmpeg_options.framerate).as_str(),
                        "-i",
                        format!("output/{}/iter-%d.png", name).as_str(),
                        "-pix_fmt",
                        "yuv420p",
                        format!("output/{}.mp4", name).as_str(),
                    ])
                    .spawn()?;
            }
            _ => (),
        };
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Space {
    density: usize,
    bounds: (Position, Position),
    holes: Vec<((Rc<RefCell<f64>>, Rc<RefCell<f64>>), (Rc<RefCell<f64>>, Rc<RefCell<f64>>))>,
    hole_multiplier: f64,
}

fn dist_rc(a: &(Rc<RefCell<f64>>, Rc<RefCell<f64>>), b: &(Rc<RefCell<f64>>, Rc<RefCell<f64>>)) -> f64 {
    dist(&(*a.0.borrow().borrow(), *a.1.borrow().borrow()), &(*b.0.borrow().borrow(), *b.1.borrow().borrow()))
}

fn dist_rc_partial(a: &(Rc<RefCell<f64>>, Rc<RefCell<f64>>), b: &(f64, f64)) -> f64 {
    dist(&(*a.0.borrow().borrow(), *a.1.borrow().borrow()), b)
}

impl Space {
    fn new_with_holes(holes: Vec<(Position, Position)>) -> Space {
        Space {
            density: 64,
            bounds: ((0.0, 0.0), (1.0, 1.0)),
            holes: holes.into_iter().map(|(((x1, y1), (x2, y2)))| ((Rc::new(RefCell::new(x1)), Rc::new(RefCell::new(y1))), (Rc::new(RefCell::new(x2)), Rc::new(RefCell::new(y2))))).collect(),
            hole_multiplier: 0.0,
        }
    }

    fn new() -> Space {
        Space::new_with_holes(Vec::new())
    }

    fn new_with_random_segment_holes(n: usize) -> Space {
        Space::new_with_holes(
            (0..n)
                .map(|_| {
                    let length = 0.02;
                    let center: (f64, f64) = (rand::random(), rand::random());
                    let angle: f64 = rand::random::<f64>() * std::f64::consts::PI * 2.0;
                    let (s, c) = angle.sin_cos();
                    let radius = (length * s, length * c);
                    let start = (center.0 + radius.0, center.1 + radius.1);
                    let end = (center.0 - radius.0, center.1 - radius.1);
                    (start, end)
                })
                .collect(),
        )
    }

    fn new_with_aligned_holes(n: usize) -> Space {
        Space::new_with_holes(
            (1..=n)
                .map(|i| {
                    let x = i as f64 / (n + 1) as f64;
                    ((x, 0.35), (x, 0.65))
                })
                .collect(),
        )
    }

    fn new_with_polyhedra_holes(n: usize, gap: f64) -> Space {
        Space::new_with_holes(
            (1..=n)
                .map(|i| {
                    let angle_start =
                        (i as f64 / n as f64) * 2.0 * std::f64::consts::PI + gap / 2.0;
                    let (s_a, c_a) = angle_start.sin_cos();
                    let angle_end =
                        ((i + 1) as f64 / n as f64) * 2.0 * std::f64::consts::PI - gap / 2.0;
                    let (s_b, c_b) = angle_end.sin_cos();
                    (
                        (0.5 + s_a / 4.0, 0.5 + c_a / 4.0),
                        (0.5 + s_b / 4.0, 0.5 + c_b / 4.0),
                    )
                })
                .collect(),
        )
    }

    fn new_with_random_holes(n: usize) -> Space {
        Space::new_with_holes((0..n).map(|_| (rand::random(), rand::random())).collect())
    }

    fn new_with_star_holes(n: usize) -> Space {
        Space::new_with_holes(
            (0..n)
                .map(|i| {
                    let angle = (i as f64 / n as f64) * std::f64::consts::PI;
                    let (s, c) = angle.sin_cos();
                    (
                        (0.5 + s / 2.0, 0.5 + c / 2.0),
                        (0.5 - s / 2.0, 0.5 - c / 2.0),
                    )
                })
                .collect(),
        )
    }

    // fn test(&self, start: &Position, end: &Position) -> f64 {
    //     let mut travel_dist = dist(start, end);
    //     for (hole_a, hole_b) in self.holes.iter() {
    //         let a_dist = dist(start, hole_a.borro);
    //         let b_dist = dist(start, hole_b);
    //         let hole_dist = if a_dist < b_dist {
    //             a_dist + dist(hole_b, end)
    //         } else {
    //             b_dist + dist(hole_a, end)
    //         } + if self.hole_multiplier > f64::default() {
    //             dist(hole_a, hole_b)
    //         } else {
    //             f64::default()
    //         };
    //         if hole_dist < travel_dist {
    //             travel_dist = hole_dist;
    //         }
    //     }
    //     travel_dist
    // }

    fn test_with_multi_travel(&self, start: &Position, end: &Position) -> f64 {
        let mut traveled = 0.0;
        let mut current_pos = start.clone();
        let mut direct = dist(&current_pos, end);
        for _ in 0..self.holes.len() + 1 {
            match self
                .holes
                .iter()
                .map(|hole| {
                    let hole_travel = if self.hole_multiplier > 0.0 {
                        dist_rc(&hole.0, &hole.1) * self.hole_multiplier
                    } else {
                        0.0
                    };
                    [
                        (
                            dist_rc_partial(&hole.0, &current_pos) + hole_travel,
                            dist_rc_partial(&hole.1, &end),
                            &hole.1,
                        ),
                        (
                            dist_rc_partial(&hole.1, &current_pos) + hole_travel,
                            dist_rc_partial(&hole.0, &end),
                            &hole.0,
                        ),
                    ]
                })
                .flatten()
                .filter(|(travel, new_dist, _)| travel + new_dist < direct)
                .min_by(|(travel_a, new_dist_a, _), (travel_b, new_dist_b, _)| {
                    (travel_a + new_dist_a)
                        .partial_cmp(&(travel_b + new_dist_b))
                        .unwrap()
                }) {
                None => {
                    traveled += direct;
                    break;
                }
                Some((travel, new_direct, new_pos)) => {
                    current_pos = (new_pos.0.borrow().borrow().clone(), new_pos.1.borrow().borrow().clone());
                    traveled += travel;
                    direct = new_direct;
                }
            }
        }
        traveled
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
                        self.test_with_multi_travel(&start, &end)
                    })
                    .sum::<f64>()
            })
            .sum();
        total / grid_density.pow(4) as f64
    }
}

impl Train for Space {
    fn render(&self) -> Option<RgbImage> {
        let dims = (1024, 1024);
        let center = ((dims.0 / 2) as f32, (dims.1 / 2) as f32);
        let max_distance = f32::hypot(center.0, center.1);
        let mut img = map_pixels(&RgbImage::new(dims.0, dims.1), |x, y, _| {
            let dist: f32 = f32::hypot(
                (x as i32 - center.0 as i32) as f32,
                (y as i32 - center.1 as i32) as f32,
            );
            let factor = (max_distance - dist) / max_distance;
            let tan = Rgb([210, 180, 140]);
            Rgb([
                (255.0 * factor + tan[0] as f32 * (1.0 - factor)) as u8,
                (255.0 * factor + tan[1] as f32 * (1.0 - factor)) as u8,
                (255.0 * factor + tan[2] as f32 * (1.0 - factor)) as u8,
            ])
        });
        draw_line_segment_mut(
            &mut img,
            (0.0, center.1),
            (dims.0 as f32, center.1),
            Rgb([220, 220, 220]),
        );
        draw_line_segment_mut(
            &mut img,
            (center.0, 0.0),
            (center.0, dims.1 as f32),
            Rgb([220, 220, 220]),
        );
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
            draw_filled_circle_mut(
                &mut img,
                (|(a, b)| (a as i32, b as i32))(start_pixel),
                2,
                Rgb([0, 0, 0]),
            );
            draw_filled_circle_mut(
                &mut img,
                (|(a, b)| (a as i32, b as i32))(end_pixel),
                2,
                Rgb([0, 0, 0]),
            );
            draw_line_segment_mut(&mut img, start_pixel, end_pixel, Rgb([128, 128, 128]));
        }
        Some(img)
    }

    fn loss(&self, seed: Option<u64>) -> f64 {
        self.permute::<ChaCha8Rng>(self.density, &mut Some(SeedableRng::seed_from_u64(seed.unwrap_or_else(|| rand::random()))))
    }

    fn get_params(&mut self) -> Vec<RefCell<f64>> {
        self.holes.iter_mut().map(|((x1, y1), (x2, y2)): ((&mut f64, &mut f64), (&mut f64, &mut f64))| [x1, y1, x2, y2]).flatten().collect()
    }
}

struct FfmpegOptions {
    framerate: usize,
}

fn slope(values: &VecDeque<f64>) -> f64 {
    let n = values.len() as f64;
    let (x, y, xy, x2) = values
        .iter()
        .enumerate()
        .map(|(i, y)| (i as f64, y))
        .fold((0.0, 0.0, 0.0, 0.0), |(sx, sy, sxy, sx2), (x, &y)| {
            (sx + x, sy + y, sxy + x * y, sx2 + x.powi(2))
        });
    let num = n * xy - x * y;
    let denom = n * x2 - x.powi(2);
    if denom == 0.0 {
        0.0
    } else {
        num / denom
    }
}

fn train_and_save(
    space: &mut Space,
    name: String,
    iters: usize,
    learn_rate: f64,
    epoch_size: usize,
    loss_window_size: usize,
    save_video: Option<FfmpegOptions>,
) -> std::io::Result<()> {
    println!("Beginning training of {}", name);
    fs::create_dir_all(format!("output/{}/", name))?;
    let mut logfile = File::create(format!("output/{}.log", name))?;
    let mut timestamp = SystemTime::now();
    let mut loss_eval_window = VecDeque::with_capacity(loss_window_size);
    for i in 0..iters {
        let (loss, seed, gradients, gradient_magnitudes) = if i == 0 {
            // default values to record the 0th iteration
            let seed = rand::random::<u64>();
            (
                space
                    .permute::<ChaCha8Rng>(epoch_size, &mut Some(SeedableRng::seed_from_u64(seed))),
                seed,
                (0..space.holes.len())
                    .map(|_| ((0.0, 0.0), (0.0, 0.0)))
                    .collect(),
                (0..(space.holes.len() * 2)).map(|_| 0.0).collect(),
            )
        } else {
            let (loss, seed, gradients) =
                space.gradient_descent(epoch_size, true, learn_rate, 1e-10, None);
            loss_eval_window.push_back(loss.log10());
            if loss_eval_window.len() > loss_window_size {
                loss_eval_window.pop_front();
            }
            let gradient_magnitudes = gradients
                .iter()
                .map(|(a, b)| [dist(a, &(0.0, 0.0)).log10(), dist(b, &(0.0, 0.0)).log10()])
                .flatten()
                .collect::<Vec<_>>();
            let new_time = SystemTime::now();
            let delta = new_time.duration_since(timestamp).unwrap();
            timestamp = new_time;
            println!(
                "Iteration {}: took: {:.4}s, loss: {:.10}, gradients: {}",
                i,
                delta.as_secs_f64(),
                loss,
                gradient_magnitudes
                    .iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            (loss, seed, gradients, gradient_magnitudes)
        };
        let loss_window_slope = slope(&loss_eval_window);
        writeln!(
            logfile,
            "{},{},{},{},{},{:?},{:?}",
            i,
            seed,
            timestamp.duration_since(UNIX_EPOCH).unwrap().as_millis(),
            loss,
            loss_window_slope,
            space.holes,
            gradients
        )?;
        if loss_eval_window.len() >= loss_window_size && loss_window_slope < 0.01
        {
            println!("Gradients settled, ending simulation");
            break;
        }
        let img = space.render();
        img.save(format!("output/{}/iter-{}.png", name, i)).unwrap();
    }
    match save_video {
        Some(ffmpeg_options) => {
            Command::new("ffmpeg")
                .args([
                    "-r",
                    format!("{}", ffmpeg_options.framerate).as_str(),
                    "-i",
                    format!("output/{}/iter-%d.png", name).as_str(),
                    "-pix_fmt",
                    "yuv420p",
                    format!("output/{}.mp4", name).as_str(),
                ])
                .spawn()?;
        }
        None => (),
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    for (mut space, name) in vec![
        (Space::new_with_polyhedra_holes(5, 0.05), "pentagon"),
    ] {
        train_and_save(
            &mut space,
            name.to_string(),
            650,
            0.1,
            64,
            100,
            Some(FfmpegOptions { framerate: 60 }),
        )?;
    }
    Ok(())
}

// fn main() -> std::io::Result<()> {
//     let mut space = Space::new_with_holes(vec![
//         (
//             (0.16889731292044485, 0.5161361052942766),
//             (0.6610321174779364, 0.7671500424093027),
//         ),
//         (
//             (0.7768302778951881, 0.8067467196950361),
//             (0.622503395204943, 0.1963612047357916),
//         ),
//         (
//             (0.8069115253149532, 0.2943328530101961),
//             (0.19300026746574028, 0.2947843252031599),
//         ),
//         (
//             (0.3766822006953432, 0.19642903936259615),
//             (0.22403880468848836, 0.8068820558817378),
//         ),
//         (
//             (0.3398511759604552, 0.7674035063258247),
//             (0.8310693248776652, 0.5147821883157463),
//         ),
//     ]);

//     let pos = (
//         &(0.45979526972906337, 0.12906427913162202),
//         &(0.4904186156070029, 0.7839874984631059),
//     );
//     let a = space.test_with_multi_travel(pos.0, pos.1);
//     space.holes[0].0 .1 += 1e-10;
//     let b = space.test_with_multi_travel(pos.0, pos.1);
//     println!("{a}, {b}");

//     // let neutral = space.permute::<ChaCha8Rng>(64, &mut Some::<ChaCha8Rng>(SeedableRng::seed_from_u64(2588624089716686143)));
//     // println!("{neutral}");
//     // space.holes[0].0.1 += 1e-10;
//     // let exception = space.permute::<ChaCha8Rng>(64, &mut Some::<ChaCha8Rng>(SeedableRng::seed_from_u64(2588624089716686143)));
//     // println!("{exception}");

//     // let results = space.gradient_descent(64, true, 0.1, 1e-10, Some(2588624089716686143));
//     // println!("{:?}", space);
//     // println!("{:?}", results);

//     Ok(())
// }
