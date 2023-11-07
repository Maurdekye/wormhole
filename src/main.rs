use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::map::map_pixels;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::collections::vec_deque;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fmt::Display;
use std::fs::File;
use std::io::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, Range, Sub};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::vec::IntoIter;
use std::{fs, iter::*};

type IFloatType = i64;
const SCALE_FACTOR: u32 = 40;
const MAX: IFloatType = IFloatType::MAX >> ((IFloatType::BITS - 1) - SCALE_FACTOR);

#[derive(Clone, Copy, Debug, Eq, Ord)]
struct IFloat {
    value: IFloatType,
}

impl IFloat {
    const MAX: IFloat = IFloat { value: MAX };

    fn hypot(self, other: Self) -> Self {
        self * self + other * other
    }

    fn sin_cos(self) -> (Self, Self) {
        let fval: f64 = self.into();
        let (s, c) = fval.sin_cos();
        (IFloat::from(s), IFloat::from(c))
    }

    fn random_with_rng<T: Rng>(rng: &mut T) -> Self {
        IFloat {
            value: rng.gen::<IFloatType>().abs() % MAX,
        }
    }

    fn random() -> Self {
        IFloat {
            value: rand::random::<IFloatType>().abs() % MAX,
        }
    }

    fn map<T>(self, f: &dyn Fn(f64) -> T) -> T {
        f(self.into())
    }

    fn pow(self, exp: usize) -> Self {
        IFloat {
            value: self.value.pow(exp as u32),
        }
    }

    fn log10(self) -> Self {
        let fval: f64 = self.into();
        IFloat::from(fval.log10())
    }

    fn abs(self) -> Self {
        IFloat {
            value: self.value.abs(),
        }
    }
}

impl From<f64> for IFloat {
    fn from(x: f64) -> IFloat {
        IFloat {
            value: (x * MAX as f64) as IFloatType,
        }
    }
}

impl From<usize> for IFloat {
    fn from(x: usize) -> IFloat {
        IFloat {
            value: ((x as f64) * MAX as f64) as IFloatType,
        }
    }
}

impl From<u32> for IFloat {
    fn from(x: u32) -> IFloat {
        IFloat {
            value: ((x as f64) * MAX as f64) as IFloatType,
        }
    }
}

impl From<i32> for IFloat {
    fn from(x: i32) -> IFloat {
        IFloat {
            value: ((x as f64) * MAX as f64) as IFloatType,
        }
    }
}

impl Into<f64> for IFloat {
    fn into(self) -> f64 {
        (self.value as f64) / MAX as f64
    }
}

impl Into<f32> for IFloat {
    fn into(self) -> f32 {
        (self.value as f32) / MAX as f32
    }
}

impl Into<i32> for IFloat {
    fn into(self) -> i32 {
        ((self.value as f32) / MAX as f32) as i32
    }
}

impl Display for IFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fval: f64 = (*self).into();
        let precision = f.precision().unwrap_or(16);
        write!(f, "{:.*}", precision, fval)?;
        Ok(())
    }
}

impl Add for IFloat {
    type Output = IFloat;

    fn add(self, rhs: Self) -> Self::Output {
        IFloat {
            value: self.value + rhs.value,
        }
    }
}

impl AddAssign for IFloat {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sub for IFloat {
    type Output = IFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        IFloat {
            value: self.value - rhs.value,
        }
    }
}

impl Mul for IFloat {
    type Output = IFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        IFloat {
            value: (((self.value as i128) * (rhs.value as i128)) >> SCALE_FACTOR) as IFloatType,
        }
    }
}

impl Div for IFloat {
    type Output = IFloat;

    fn div(self, rhs: Self) -> Self::Output {
        IFloat {
            value: (((self.value as i128) << SCALE_FACTOR) / rhs.value as i128) as IFloatType,
        }
    }
}

impl PartialEq for IFloat {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for IFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl std::iter::Sum for IFloat {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or(ZERO.clone())
    }
}

type Position = (IFloat, IFloat);

const PI: IFloat = IFloat {
    value: (std::f64::consts::PI * MAX as f64) as IFloatType,
};
const ZERO: IFloat = IFloat {
    value: (0.0 * MAX as f64) as IFloatType,
};

fn dist(a: &Position, b: &Position) -> IFloat {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.hypot(dy)
    // dx.abs() + dy.abs()
}

#[derive(Clone)]
struct Bitmask {
    mask: u64,
    capacity: usize,
}

impl Bitmask {
    fn with_capacity(capacity: usize) -> Bitmask {
        Bitmask {
            mask: 0,
            capacity: capacity,
        }
    }

    fn new() -> Bitmask {
        Self::with_capacity(u32::BITS as usize)
    }

    fn contains(&self, value: usize) -> bool {
        self.mask & (1 << value) != 0
    }

    fn into_set(self, value: usize) -> Self {
        Bitmask {
            mask: self.mask | (1u64 << value),
            capacity: self.capacity,
        }
    }

    fn into_unset(self, value: usize) -> Self {
        Bitmask {
            mask: self.mask & !(1u64 << value),
            capacity: self.capacity,
        }
    }

    fn iter(&self) -> BitmaskIter {
        BitmaskIter {
            mask: self.mask,
            capacity: self.capacity,
            position: 0,
            pos_bit: 1,
        }
    }
}

impl From<Range<usize>> for Bitmask {
    fn from(value: Range<usize>) -> Self {
        let mut mask = Bitmask::with_capacity(value.end);
        for i in value {
            mask.mask |= 1u64 << i;
        }
        mask
    }
}

struct BitmaskIter {
    mask: u64,
    capacity: usize,
    position: usize,
    pos_bit: u64,
}

impl Iterator for BitmaskIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.capacity {
            return None;
        }
        while self.mask & self.pos_bit == 0 {
            self.position += 1;
            self.pos_bit <<= 1;
            if self.position >= self.capacity {
                return None;
            }
        }
        self.position += 1;
        self.pos_bit <<= 1;
        Some(self.position - 1)
    }
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

#[derive(Clone, Debug)]
struct Space {
    bounds: (Position, Position),
    holes: Vec<Position>,
}

impl Space {
    fn new_with_holes(holes: Vec<Position>) -> Space {
        Space {
            bounds: (
                (IFloat::from(ZERO.clone()), IFloat::from(ZERO.clone())),
                (IFloat::from(1.0), IFloat::from(1.0)),
            ),
            holes: holes,
        }
    }

    fn new() -> Space {
        Space::new_with_holes(Vec::new())
    }

    fn new_with_aligned_holes(n: usize) -> Space {
        Space::new_with_holes(
            (1..=n)
                .map(|i| {
                    let x = IFloat::from(i) / IFloat::from(n + 1);
                    (x, IFloat::from(0.5))
                })
                .collect(),
        )
    }

    fn new_with_polyhedra_holes(n: usize, gap: IFloat) -> Space {
        Space::new_with_holes(
            (1..=n)
                .map(|i| {
                    let angle_start = (IFloat::from(i) / IFloat::from(n)) * IFloat::from(2.0) * PI
                        + gap / IFloat::from(2.0);
                    let (s, c) = angle_start.sin_cos();
                    (
                        IFloat::from(0.5) + s / IFloat::from(4.0),
                        IFloat::from(0.5) + c / IFloat::from(4.0),
                    )
                })
                .collect(),
        )
    }

    fn new_with_random_holes(n: usize) -> Space {
        Space::new_with_holes(
            (0..n)
                .map(|_| (IFloat::random(), IFloat::random()))
                .collect(),
        )
    }

    fn exhaustive_test(&self, start: &Position, end: &Position) -> IFloat {
        let direct_distance = dist(start, end);
        let mut holes = self.holes.iter();
        let first_hole = holes.next().unwrap();
        let mut closest_entrance_hole_distance = dist(start, first_hole);
        let mut closest_exit_hole_distance = dist(end, first_hole);
        for hole in holes {
            let entrance_distance = dist(hole, start);
            if entrance_distance < closest_entrance_hole_distance {
                closest_entrance_hole_distance = entrance_distance;
            }
            let exit_distance = dist(hole, end);
            if exit_distance < closest_exit_hole_distance {
                closest_exit_hole_distance = exit_distance;
            }
        }
        direct_distance.min(closest_entrance_hole_distance + closest_exit_hole_distance)
    }

    fn permute<T>(&self, grid_density: usize, random_placement: &mut Option<T>) -> IFloat
    where
        T: Rng,
    {
        let float_density = IFloat::from(grid_density);
        let cell_width = (self.bounds.1 .0 - self.bounds.0 .0) / float_density;
        let cell_height = (self.bounds.1 .1 - self.bounds.0 .1) / float_density;
        let offsets = GridIter::between(&(0, 0), &(grid_density, grid_density))
            .map(|_| match random_placement {
                None => (
                    cell_width / IFloat::from(2.0),
                    cell_height / IFloat::from(2.0),
                ),
                Some(rng) => (
                    IFloat::random_with_rng(rng) * cell_width,
                    IFloat::random_with_rng(rng) * cell_height,
                ),
            })
            .collect::<Vec<Position>>();
        let mut total: IFloat = GridIter::between(&(0, 0), &(grid_density, grid_density))
            .collect::<Vec<_>>()
            .par_iter()
            .enumerate()
            .map(|(si, (start_x, start_y))| {
                let start = (
                    self.bounds.0 .0 + cell_width * IFloat::from(*start_x) + offsets[si].0,
                    self.bounds.0 .1 + cell_height * IFloat::from(*start_y) + offsets[si].1,
                );
                GridIter::between(&(0, 0), &(grid_density, grid_density))
                    .enumerate()
                    .map(|(ei, (end_x, end_y))| {
                        let end = (
                            self.bounds.0 .0 + cell_width * IFloat::from(end_x) + offsets[ei].0,
                            self.bounds.0 .1 + cell_height * IFloat::from(end_y) + offsets[ei].1,
                        );
                        self.exhaustive_test(&start, &end)
                    })
                    .sum::<IFloat>()
            })
            .sum();
        for _ in 0..4 {
            total = total / float_density;
        }
        total
    }

    fn gradient_descent(
        &mut self,
        density: usize,
        random_placement: bool,
        temperature: IFloat,
        epsilon: IFloat,
        seed: Option<u64>,
    ) -> (IFloat, u64, Vec<Position>) {
        let mut otherself = self.clone();
        let seed = seed.unwrap_or_else(|| rand::random::<u64>());
        let get_rng = || {
            if random_placement {
                Some::<ChaCha8Rng>(SeedableRng::seed_from_u64(seed))
            } else {
                None
            }
        };
        let neutral = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
        // println!("neutral:  {neutral}");
        let gradients = self
            .holes
            .iter_mut()
            .enumerate()
            .map(|(i, hole)| {
                let mut h = |l: &dyn Fn(&mut (IFloat, IFloat)) -> &mut IFloat| {
                    *l(&mut otherself.holes[i]) = *l(hole) + epsilon;
                    let gradient = otherself.permute::<ChaCha8Rng>(density, &mut get_rng());
                    // println!("gradient: {gradient}");
                    *l(&mut otherself.holes[i]) = *l(hole);
                    gradient
                };

                let g = |gradient| {
                    if neutral == gradient {
                        ZERO.clone()
                    } else {
                        (neutral - gradient) / epsilon
                    }
                };

                let f = |gradient_x, gradient_y| {
                    let (gradient_x, gradient_y) = (g(gradient_x), g(gradient_y));
                    let magnitude = gradient_x.hypot(gradient_y);
                    if magnitude > temperature {
                        (
                            temperature * gradient_x / magnitude,
                            temperature * gradient_y / magnitude,
                        )
                    } else {
                        (gradient_x, gradient_y)
                    }
                };

                (f(h(&|x| &mut x.0), h(&|x| &mut x.1)))
            })
            .collect::<Vec<_>>();
        (
            neutral,
            seed,
            gradients
                .into_iter()
                .zip(self.holes.iter_mut())
                .map(|(gradient, hole)| {
                    hole.0 += gradient.0 * temperature;
                    hole.1 += gradient.1 * temperature;
                    gradient
                })
                .collect(),
        )
    }

    fn render(&self) -> RgbImage {
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
            let (bound_width, bound_height) = (
                self.bounds.1 .0 - self.bounds.0 .0,
                self.bounds.1 .1 - self.bounds.0 .1,
            );
            let hole_pixel: (i32, i32) = (
                (((hole.0 - self.bounds.0 .0) / bound_width) * IFloat::from(dims.0)).into(),
                (((hole.1 - self.bounds.0 .1) / bound_height) * IFloat::from(dims.1)).into(),
            );
            draw_filled_circle_mut(
                &mut img,
                (|(a, b)| (a as i32, b as i32))(hole_pixel),
                3,
                Rgb([0, 0, 0]),
            );
        }
        img
    }
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

struct FfmpegSettings {
    framerate: usize,
}

struct LearnSettings {
    training_epsilon: f64,
    loss_slope_epsilon: f64,
    init_learn_rate: f64,
    learn_rate_reductions: usize,
    learn_rate_reduction_ratio: f64,
    loss_window_size: usize,
    max_iters: Option<usize>,
    epoch_size: usize,
}

fn train_and_save(
    space: &mut Space,
    name: String,
    learn_settings: LearnSettings,
    save_video: Option<FfmpegSettings>,
) -> std::io::Result<()> {
    println!("Beginning training of {}", name);

    fs::create_dir_all(format!("output/{}/", name))?;
    let mut logfile = File::create(format!("output/{}.log", name))?;

    let mut timestamp = SystemTime::now();
    let mut loss_eval_window: VecDeque<f64> =
        VecDeque::with_capacity(learn_settings.loss_window_size + 1);
    let mut learn_reductions = 0;
    let mut current_learn_rate = learn_settings.init_learn_rate;

    for i in 0.. {
        if learn_settings
            .max_iters
            .map(|iters| i > iters)
            .unwrap_or(false)
        {
            break;
        }

        let (loss, seed, gradients) = if i == 0 {
            // default values to record the 0th iteration
            let seed = rand::random::<u64>();
            (
                space.permute::<ChaCha8Rng>(
                    learn_settings.epoch_size,
                    &mut Some(SeedableRng::seed_from_u64(seed)),
                ),
                seed,
                (0..space.holes.len())
                    .map(|_| (ZERO.clone(), ZERO.clone()))
                    .collect(),
            )
        } else {
            let (loss, seed, gradients) = space.gradient_descent(
                learn_settings.epoch_size,
                true,
                current_learn_rate.into(),
                IFloat::from(learn_settings.training_epsilon),
                None,
            );

            loss_eval_window.push_back(Into::<f64>::into(loss).log10());
            if loss_eval_window.len() > learn_settings.loss_window_size {
                loss_eval_window.pop_front();
            }

            let new_time = SystemTime::now();
            let delta = new_time.duration_since(timestamp).unwrap();
            timestamp = new_time;

            let gradient_magnitudes = gradients
                .iter()
                .map(|g| dist(g, &(ZERO.clone(), ZERO.clone())).log10())
                .collect::<Vec<_>>();

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
            space.holes,
            gradients
        )?;

        let img = space.render();
        img.save(format!("output/{}/iter-{}.png", name, i)).unwrap();

        if loss_eval_window.len() >= learn_settings.loss_window_size
            && loss_window_slope.abs() < learn_settings.loss_slope_epsilon
        {
            if learn_reductions < learn_settings.learn_rate_reductions {
                current_learn_rate *= learn_settings.learn_rate_reduction_ratio;
                learn_reductions += 1;
                loss_eval_window.clear();
                println!(
                    "Reducing learning rate to {current_learn_rate} {learn_reductions}/{} times",
                    learn_settings.learn_rate_reductions
                );
            } else {
                println!("Gradients settled, ending simulation");
                break;
            }
        }
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
                    "-y",
                    format!("output/{}.mp4", name).as_str(),
                ])
                .spawn()?;
        }
        None => (),
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    for i in 32..=32 {
        let mut space = Space::new_with_random_holes(i);
        train_and_save(
            &mut space,
            format!("{i}_holes_2"),
            LearnSettings {
                training_epsilon: 1e-8,
                loss_slope_epsilon: 1e-7,
                init_learn_rate: 10.0,
                learn_rate_reductions: 2,
                learn_rate_reduction_ratio: 0.1,
                loss_window_size: 100,
                max_iters: None,
                epoch_size: 64,
            },
            Some(FfmpegSettings { framerate: 60 }),
        )?;
    }
    Ok(())
}

// fn main() -> std::io::Result<()> {
//     for (mut space, name) in vec![
//         (Space::new_with_random_holes(13), "13_holes"),
//         (Space::new_with_random_holes(14), "14_holes"),
//         (Space::new_with_random_holes(15), "15_holes"),
//         (Space::new_with_random_holes(16), "16_holes"),
//         (Space::new_with_random_holes(17), "17_holes"),
//         (Space::new_with_random_holes(18), "18_holes"),
//         (Space::new_with_random_holes(19), "19_holes"),
//         (Space::new_with_random_holes(20), "20_holes"),
//         (Space::new_with_random_holes(32), "32_holes"),
//     ] {
//         train_and_save(
//             &mut space,
//             name.to_string(),
//             None,
//             1.0,
//             2,
//             64,
//             100,
//             Some(FfmpegOptions { framerate: 60 }),
//         )?;
//     }
//     Ok(())
// }

// fn main() {
//     let space = Space::new_with_aligned_holes(3);
//     // space.exhaustive_test(&(0.1.into(), 0.1.into()), &(0.9.into(), 0.9.into()));
//     let mut logfile = File::create(format!("output/opt5_perf.log")).unwrap();
//     for i in 32..=64 {
//         let start = SystemTime::now();
//         let result = space.permute::<ChaCha8Rng>(i, &mut None);
//         let duration = SystemTime::now().duration_since(start).unwrap();
//         println!("{i}: {:.3}s, {result:.6}", duration.as_secs_f64());
//         writeln!(logfile, "{i},{},{result}", duration.as_millis()).unwrap();
//     }
// }

// fn main() {
//     let mut logfile = File::create(format!("output/holecount_perf.log")).unwrap();
//     for i in 1..=8 {
//         let space = Space::new_with_aligned_holes(i);
//         let start = SystemTime::now();
//         let result = space.permute::<ChaCha8Rng>(64, &mut None);
//         let duration = SystemTime::now().duration_since(start).unwrap();
//         println!("{i}: {:.3}s, {result:.6}", duration.as_secs_f64());
//         writeln!(logfile, "{i},{},{result}", duration.as_millis()).unwrap();
//     }
// }

// fn main() -> std::io::Result<()> {
//     for (mut space, name) in vec![
//         // (Space::new_with_star_holes(4), "asterisk_2"),
//         (Space::new_with_random_segment_holes(3), "triple_3"),
//         // (Space::new_with_random_segment_holes(5), "quintouple"),
//         // (Space::new_with_random_segment_holes(6), "sextouple"),
//         // (Space::new_with_random_segment_holes(7), "septouble"),
//         // (Space::new_with_random_segment_holes(8), "octouple"),
//     ] {
//         train_and_save(
//             &mut space,
//             name.to_string(),
//             None,
//             0.1,
//             64,
//             200,
//             Some(FfmpegOptions { framerate: 60 }),
//         )?;
//     }
//     Ok(())
// }

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
