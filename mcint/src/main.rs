use rand::prelude::*;
use mcint::{D, N, L, atomicsystem_ft, atomicsystem_int};

fn main() {
    // set a random number generator
    let mut rng = rand::thread_rng();
    
    // create position allocation and fill randomly
    let mut pos: [[f64; D]; N]= [[0.0; D]; N];
    for array in pos.iter_mut() {
        for elem in array.iter_mut() {
            *elem = rng.gen_range(0.0..L);
        }
    }
    let e_tot2 = atomicsystem_ft(pos);
    // create position allocation and fill randomly
    let mut pos: [[isize; D]; N] = [[0; D]; N];

    for array in pos.iter_mut() {
        for elem in array.iter_mut() {
            *elem = rng.gen::<isize>();
        }
    }
    let e_tot1 = atomicsystem_int(pos);
    
    println!("{e_tot1:7.5e}, {e_tot2:7.5e}")
}
