pub const D: usize = 3;
pub const N: usize = 1000;
pub const L: f64 = 30.0;
pub const DX: f64 = L / (u32::MAX as f64);

pub fn atomicsystem_ft(pos: [[f64; D]; N]) -> f64 {
  let mut dx: f64;
  let mut r2 = 0.0;
  let mut sr6: f64;
  let mut sr12: f64;
  let mut e_tot: f64 = 0.0;
  for (i, pos_i) in pos.iter().enumerate() {
    for pos_j in pos.iter().skip(i + 1) {
      for k in 0..D {
        dx = pos_j[k] - pos_i[k];
        dx = dx - L * (dx / L).round();
        r2 += dx * dx;
      }
      sr6 = 1.0 / (r2.powi(3));
      sr12 = sr6.powi(2);
      e_tot += sr12 - sr6;
    }
  }
  return e_tot;
}

pub fn atomicsystem_int(pos: [[isize; D]; N]) -> f64 {
  let mut dx: isize;
  let mut dxf64: f64;
  let mut r2 = 0.0;
  let mut sr6: f64;
  let mut sr12: f64;
  let mut e_tot = 0.0;
  for (i, pos_i) in pos.iter().enumerate() {
    for pos_j in pos.iter().skip(i + 1) {
      for k in 0..D {
        dx = pos_j[k].wrapping_sub(pos_i[k]);
        dxf64 = (dx as f64) * DX;
        r2 += dxf64 * dxf64;
      }
      sr6 = 1.0 / (r2.powi(3));
      sr12 = sr6.powi(2);
      e_tot += sr12 - sr6;
    }
  }
  return e_tot;
}
