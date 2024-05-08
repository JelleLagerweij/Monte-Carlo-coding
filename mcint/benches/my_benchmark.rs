use mcint::{D, N, L, atomicsystem_ft, atomicsystem_int};
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use rand::prelude::*;

const N_SAMPLES: usize = 1000;


fn benchmark_ft(c: &mut Criterion){
    // set a random number generator
    let mut rng = rand::thread_rng();
    
    // create position allocation and fill randomly
    let mut pos: [[f64; D]; N]= [[0.0; D]; N];
    for array in pos.iter_mut() {
        for elem in array.iter_mut() {
            *elem = rng.gen_range(0.0..L);
        }
    }

    let mut group = c.benchmark_group("Floating point benchmark group");
    group.sample_size(N_SAMPLES); // Increase the number of samples
    group.bench_function("testing with float64", |b| b.iter(|| atomicsystem_ft(black_box(pos))));
    group.finish();// Enable flat sampling
}

fn benchmark_int(c: &mut Criterion){
    // set a random number generator
    let mut rng = rand::thread_rng();

    // create position allocation and fill randomly
    let mut pos: [[isize; D]; N] = [[0; D]; N];

    for array in pos.iter_mut() {
        for elem in array.iter_mut() {
            *elem = rng.gen::<isize>();
        }
    }

    let mut group = c.benchmark_group("Integer point benchmark group");
    group.sample_size(N_SAMPLES); // Increase the number of samples
    group.bench_function("testing with int", |b| b.iter(|| atomicsystem_int(black_box(pos))));
    group.finish();// Enable flat sampling
}

criterion_group!(benches, benchmark_ft, benchmark_int);
criterion_main!(benches); // Increase target time to 7 seconds

