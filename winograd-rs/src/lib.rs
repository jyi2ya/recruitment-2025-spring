#![allow(dead_code, non_snake_case, unused_assignments)]
use core::mem::size_of;

#[derive(Copy, Clone)]
struct TileIndex {
    pub b: i64,
    pub th: i64,
    pub tw: i64,
}
#[derive(Copy, Clone)]
struct FilterShape {
    pub oc: i64,
    pub ic: i64,
    pub h: i64,
    pub w: i64,
}
#[derive(Copy, Clone)]
struct ImageShape {
    pub bs: i64,
    pub ic: i64,
    pub h: i64,
    pub w: i64,
}
#[derive(Copy, Clone)]
struct UShape {
    pub h: i64,
    pub w: i64,
    pub oc: i64,
    pub ic: i64,
}
#[derive(Copy, Clone)]
struct VShape {
    pub h: i64,
    pub w: i64,
    pub num_tiles: i64,
    pub ic: i64,
}
#[derive(Copy, Clone)]
struct OutShape {
    pub bs: i64,
    pub oc: i64,
    pub h: i64,
    pub w: i64,
}
#[derive(Copy, Clone)]
struct TilingInfo {
    pub bs: i64,
    pub num_tile_per_image: i64,
    pub num_tiles: i64,
    pub tiles_on_h: i64,
    pub tiles_on_w: i64,
    pub tile_in_h: i64,
    pub tile_in_w: i64,
    pub tile_out_h: i64,
    pub tile_out_w: i64,
}

#[inline]
fn get_output_shape(is: ImageShape, fs: FilterShape) -> OutShape {
    OutShape {
        bs: is.bs,
        oc: fs.oc,
        h: is.h - fs.h + 1,
        w: is.w - fs.w + 1,
    }
}

#[inline]
fn get_tiling_info(is: ImageShape, os: OutShape) -> TilingInfo {
    let tiles_on_h = (os.h + 4 - 1) / 4;
    let tiles_on_w = (os.w + 4 - 1) / 4;
    let num_tile_per_image = tiles_on_h * tiles_on_w;
    TilingInfo {
        tiles_on_h,
        tiles_on_w,
        bs: is.bs,
        num_tile_per_image,
        num_tiles: num_tile_per_image * is.bs,
        tile_in_h: 6,
        tile_in_w: 6,
        tile_out_h: 4,
        tile_out_w: 4,
    }
}

#[inline]
fn get_U_shape(fs: FilterShape, ti: TilingInfo) -> UShape {
    UShape {
        oc: fs.oc,
        ic: fs.ic,
        h: ti.tile_in_h,
        w: ti.tile_in_w,
    }
}

#[inline]
fn get_V_shape(is: ImageShape, ti: TilingInfo) -> VShape {
    VShape {
        num_tiles: ti.num_tiles,
        ic: is.ic,
        h: ti.tile_in_h,
        w: ti.tile_in_w,
    }
}

#[inline]
fn get_tile_index(tile: i64, ts: TilingInfo) -> TileIndex {
    TileIndex {
        b: tile / ts.num_tile_per_image,
        th: tile % ts.num_tile_per_image / ts.tiles_on_w,
        tw: tile % ts.num_tile_per_image % ts.tiles_on_w,
    }
}

fn image_transform_h(V: &mut [f32], vs: VShape, ti: TilingInfo, collapsed_dim_size: i64) {
    for h in 0..ti.tile_in_h {
        for idx in 0..collapsed_dim_size {
            let load = V[(h * vs.w * collapsed_dim_size + 0 * collapsed_dim_size + idx) as usize];
            let mut z0 = 4.0f32 * load;

            let load = V[(h * vs.w * collapsed_dim_size + 1 * collapsed_dim_size + idx) as usize];
            let mut z1 = -4.0f32 * load;
            let mut z2 = 4.0f32 * load;
            let mut z3 = -2.0f32 * load;
            let mut z4 = 2.0f32 * load;
            let mut z5 = 4.0f32 * load;

            let load = V[(h * vs.w * collapsed_dim_size + 2 * collapsed_dim_size + idx) as usize];
            z0 += -5.0f32 * load;
            z1 += -4.0f32 * load;
            z2 += -4.0f32 * load;
            z3 += -load;
            z4 += -load;

            let load = V[(h * vs.w * collapsed_dim_size + 3 * collapsed_dim_size + idx) as usize];
            z1 += load;
            z2 += -load;
            z3 += 2.0f32 * load;
            z4 += -2.0f32 * load;
            z5 += -5.0f32 * load;

            let load = V[(h * vs.w * collapsed_dim_size + 4 * collapsed_dim_size + idx) as usize];
            z0 += load;
            z1 += load;
            z2 += load;
            z3 += load;
            z4 += load;

            let load = V[(h * vs.w * collapsed_dim_size + 5 * collapsed_dim_size + idx) as usize];
            z5 += load;

            V[(h * vs.w * collapsed_dim_size + 0 * collapsed_dim_size + idx) as usize] = z0;
            V[(h * vs.w * collapsed_dim_size + 1 * collapsed_dim_size + idx) as usize] = z1;
            V[(h * vs.w * collapsed_dim_size + 2 * collapsed_dim_size + idx) as usize] = z2;
            V[(h * vs.w * collapsed_dim_size + 3 * collapsed_dim_size + idx) as usize] = z3;
            V[(h * vs.w * collapsed_dim_size + 4 * collapsed_dim_size + idx) as usize] = z4;
            V[(h * vs.w * collapsed_dim_size + 5 * collapsed_dim_size + idx) as usize] = z5;
        }
    }
}

fn image_transform_w(
    packed_image: &[f32],
    V: &mut [f32],
    vs: VShape,
    ti: TilingInfo,
    collapsed_dim_size: i64,
) {
    for w in 0..ti.tile_in_w {
        for idx in 0..collapsed_dim_size {
            let load = packed_image
                [(0 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            let mut z0 = 4.0f32 * load;

            let load = packed_image
                [(1 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            let mut z1 = -4.0f32 * load;
            let mut z2 = 4.0f32 * load;
            let mut z3 = -2.0f32 * load;
            let mut z4 = 2.0f32 * load;
            let mut z5 = 4.0f32 * load;

            let load = packed_image
                [(2 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z0 += -5.0f32 * load;
            z1 += -4.0f32 * load;
            z2 += -4.0f32 * load;
            z3 += -load;
            z4 += -load;

            let load = packed_image
                [(3 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z1 += load;
            z2 += -load;
            z3 += 2.0f32 * load;
            z4 += -2.0f32 * load;
            z5 += -5.0f32 * load;

            let load = packed_image
                [(4 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z0 += load;
            z1 += load;
            z2 += load;
            z3 += load;
            z4 += load;

            let load = packed_image
                [(5 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z5 += load;

            V[(0 * vs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z0;
            V[(1 * vs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z1;
            V[(2 * vs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z2;
            V[(3 * vs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z3;
            V[(4 * vs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z4;
            V[(5 * vs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z5;
        }
    }
}

fn filter_transform_w(
    packed_filter: &[f32],
    U: &mut [f32],
    fs: FilterShape,
    us: UShape,
    collapsed_dim_size: i64,
) {
    for w in 0..fs.w {
        for idx in 0..collapsed_dim_size {
            let z6 = packed_filter
                [(0 * fs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            let z0 = 1.0f32 / 4.0f32 * z6;
            let mut z1 = -1.0f32 / 6.0f32 * z6;
            let mut z2 = -1.0f32 / 6.0f32 * z6;
            let mut z3 = 1.0f32 / 24.0f32 * z6;
            let mut z4 = 1.0f32 / 24.0f32 * z6;
            let z6 = packed_filter
                [(1 * fs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z1 += -1.0f32 / 6.0f32 * z6;
            z2 += 1.0f32 / 6.0f32 * z6;
            z3 += 1.0f32 / 12.0f32 * z6;
            z4 += -1.0f32 / 12.0f32 * z6;
            let z6 = packed_filter
                [(2 * fs.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z1 += -1.0f32 / 6.0f32 * z6;
            z2 += -1.0f32 / 6.0f32 * z6;
            z3 += 1.0f32 / 6.0f32 * z6;
            z4 += 1.0f32 / 6.0f32 * z6;
            let z5 = z6;
            U[(0 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z0;
            U[(1 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z1;
            U[(2 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z2;
            U[(3 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z3;
            U[(4 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z4;
            U[(5 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z5;
        }
    }
}

fn filter_transform_h(U: &mut [f32], _fs: FilterShape, us: UShape, collapsed_dim_size: i64) {
    for h in 0..us.h {
        for idx in 0..collapsed_dim_size {
            let z6 = U[(h * us.w * collapsed_dim_size + 0 * collapsed_dim_size + idx) as usize];
            let z0 = 1.0f32 / 4.0f32 * z6;
            let mut z1 = -1.0f32 / 6.0f32 * z6;
            let mut z2 = -1.0f32 / 6.0f32 * z6;
            let mut z3 = 1.0f32 / 24.0f32 * z6;
            let mut z4 = 1.0f32 / 24.0f32 * z6;
            let z6 = U[(h * us.w * collapsed_dim_size + 1 * collapsed_dim_size + idx) as usize];
            z1 += -1.0f32 / 6.0f32 * z6;
            z2 += 1.0f32 / 6.0f32 * z6;
            z3 += 1.0f32 / 12.0f32 * z6;
            z4 += -1.0f32 / 12.0f32 * z6;
            let z6 = U[(h * us.w * collapsed_dim_size + 2 * collapsed_dim_size + idx) as usize];
            z1 += -1.0f32 / 6.0f32 * z6;
            z2 += -1.0f32 / 6.0f32 * z6;
            z3 += 1.0f32 / 6.0f32 * z6;
            z4 += 1.0f32 / 6.0f32 * z6;
            let z5 = z6;
            U[(h * us.w * collapsed_dim_size + 0 * collapsed_dim_size + idx) as usize] = z0;
            U[(h * us.w * collapsed_dim_size + 1 * collapsed_dim_size + idx) as usize] = z1;
            U[(h * us.w * collapsed_dim_size + 2 * collapsed_dim_size + idx) as usize] = z2;
            U[(h * us.w * collapsed_dim_size + 3 * collapsed_dim_size + idx) as usize] = z3;
            U[(h * us.w * collapsed_dim_size + 4 * collapsed_dim_size + idx) as usize] = z4;
            U[(h * us.w * collapsed_dim_size + 5 * collapsed_dim_size + idx) as usize] = z5;
        }
    }
}

fn output_transform_w(M: &[f32], Y: &mut [f32], ti: TilingInfo, collapsed_dim_size: i64) {
    for w in 0..ti.tile_in_w {
        for idx in 0..collapsed_dim_size {
            let z4 =
                M[(0 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            let mut z0 = z4;
            let z4 =
                M[(1 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z0 += z4;
            let mut z1 = z4;
            let mut z2 = z4;
            let mut z3 = z4;
            let z4 =
                M[(2 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z0 += z4;
            z1 += -z4;
            z2 += z4;
            z3 += -z4;
            let z4 =
                M[(3 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z0 += z4;
            z1 += 2.0f32 * z4;
            z2 += 4.0f32 * z4;
            z3 += 8.0f32 * z4;
            let z4 =
                M[(4 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z0 += z4;
            z1 += -2.0f32 * z4;
            z2 += 4.0f32 * z4;
            z3 += -8.0f32 * z4;
            let z4 =
                M[(5 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize];
            z3 += z4;
            Y[(0 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z0;
            Y[(1 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z1;
            Y[(2 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z2;
            Y[(3 * ti.tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx) as usize] = z3;
        }
    }
}

fn output_transform_h(Y: &mut [f32], ti: TilingInfo, collapsed_dim_size: i64) {
    for h in 0..ti.tile_out_h {
        for idx in 0..collapsed_dim_size {
            let z4 =
                Y[(h * ti.tile_in_w * collapsed_dim_size + 0 * collapsed_dim_size + idx) as usize];
            let mut z0 = z4;
            let z4 =
                Y[(h * ti.tile_in_w * collapsed_dim_size + 1 * collapsed_dim_size + idx) as usize];
            z0 += z4;
            let mut z1 = z4;
            let mut z2 = z4;
            let mut z3 = z4;
            let z4 =
                Y[(h * ti.tile_in_w * collapsed_dim_size + 2 * collapsed_dim_size + idx) as usize];
            z0 += z4;
            z1 += -z4;
            z2 += z4;
            z3 += -z4;
            let z4 =
                Y[(h * ti.tile_in_w * collapsed_dim_size + 3 * collapsed_dim_size + idx) as usize];
            z0 += z4;
            z1 += 2.0f32 * z4;
            z2 += 4.0f32 * z4;
            z3 += 8.0f32 * z4;
            let z4 =
                Y[(h * ti.tile_in_w * collapsed_dim_size + 4 * collapsed_dim_size + idx) as usize];
            z0 += z4;
            z1 += -2.0f32 * z4;
            z2 += 4.0f32 * z4;
            z3 += -8.0f32 * z4;
            let z4 =
                Y[(h * ti.tile_in_w * collapsed_dim_size + 5 * collapsed_dim_size + idx) as usize];
            z3 += z4;
            Y[(h * ti.tile_in_w * collapsed_dim_size + 0 * collapsed_dim_size + idx) as usize] = z0;
            Y[(h * ti.tile_in_w * collapsed_dim_size + 1 * collapsed_dim_size + idx) as usize] = z1;
            Y[(h * ti.tile_in_w * collapsed_dim_size + 2 * collapsed_dim_size + idx) as usize] = z2;
            Y[(h * ti.tile_in_w * collapsed_dim_size + 3 * collapsed_dim_size + idx) as usize] = z3;
        }
    }
}

fn filter_packing(filter: &mut [f32], packed_filter: &mut [f32], fs: FilterShape) {
    for h in 0..fs.h {
        for w in 0..fs.w {
            for oc in 0..fs.oc {
                for ic in 0..fs.ic {
                    packed_filter[(h * fs.w * fs.oc * fs.ic + w * fs.oc * fs.ic + oc * fs.ic + ic)
                        as usize] = filter
                        [(oc * fs.ic * fs.h * fs.w + ic * fs.h * fs.w + h * fs.w + w) as usize];
                }
            }
        }
    }
}

fn image_packing(image: &mut [f32], packed_image: &mut [f32], is: ImageShape, ti: TilingInfo) {
    for h in 0..ti.tile_in_h {
        for w in 0..ti.tile_in_w {
            for tile in 0..ti.num_tiles {
                for ic in 0..is.ic {
                    let tidx: TileIndex = get_tile_index(tile, ti);
                    let batch: i64 = tidx.b;
                    let ww: i64 = tidx.tw;
                    let hh: i64 = tidx.th;
                    packed_image[(h * ti.tile_in_w * ti.num_tiles * is.ic
                        + w * ti.num_tiles * is.ic
                        + tile * is.ic
                        + ic) as usize] = if hh * 4 + h < is.h && ww * 4 + w < is.w {
                        image[(batch * is.ic * is.h * is.w
                            + ic * is.h * is.w
                            + (hh * 4 + h) * is.w
                            + (ww * 4 + w)) as usize]
                    } else {
                        0.
                    };
                }
            }
        }
    }
}

fn output_unpacking_store(Y: &[f32], out: &mut [f32], os: OutShape, ti: TilingInfo) {
    for h in 0..ti.tile_out_h {
        for w in 0..ti.tile_out_w {
            for oc in 0..os.oc {
                for tile in 0..ti.num_tiles {
                    let tidx: TileIndex = get_tile_index(tile, ti);
                    let batch: i64 = tidx.b;
                    let ww: i64 = tidx.tw;
                    let hh: i64 = tidx.th;
                    if hh * 4 + h < os.h && ww * 4 + w < os.w {
                        out[(batch * os.oc * os.h * os.w
                            + oc * os.h * os.w
                            + (hh * 4 + h) * os.w
                            + (ww * 4 + w)) as usize] = Y[(h * ti.tile_in_w * os.oc * ti.num_tiles
                            + w * os.oc * ti.num_tiles
                            + oc * ti.num_tiles
                            + tile)
                            as usize];
                    }
                }
            }
        }
    }
}

fn sgemm(M: i64, N: i64, K: i64, A: &[f32], B: &[f32], C: &mut [f32]) {
    for m in 0..M {
        for n in 0..N {
            C[(n * M + m) as usize] = 0.;
            for k in 0..K {
                C[(n * M + m) as usize] += A[(m * K + k) as usize] * B[(n * K + k) as usize];
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn winograd_convolution(
    image: *mut libc::c_float,
    image_height: libc::c_int,
    image_width: libc::c_int,
    input_channel_num: libc::c_int,
    filter: *mut libc::c_float,
    output_channel_num: libc::c_int,
    batch_num: libc::c_int,
    out: *mut libc::c_float,
) {
    let is = ImageShape {
        bs: batch_num as i64,
        ic: input_channel_num as i64,
        h: image_height as i64,
        w: image_width as i64,
    };
    let image =
        unsafe { std::slice::from_raw_parts_mut(image, (is.bs * is.h * is.w * is.ic) as usize) };

    let fs = FilterShape {
        oc: output_channel_num as i64,
        ic: input_channel_num as i64,
        h: 3 as i64,
        w: 3 as i64,
    };
    let filter =
        unsafe { std::slice::from_raw_parts_mut(filter, (fs.ic * fs.h * fs.w * fs.oc) as usize) };

    let os: OutShape = get_output_shape(is, fs);
    let out =
        unsafe { std::slice::from_raw_parts_mut(out, (os.bs * os.h * os.w * os.oc) as usize) };

    let ti: TilingInfo = get_tiling_info(is, os);
    let us: UShape = get_U_shape(fs, ti);
    let vs: VShape = get_V_shape(is, ti);
    let mut packed_filter =
        vec![0.; (size_of::<f32>() as i64 * fs.h * fs.w * fs.oc * fs.ic) as usize];
    let mut packed_image = vec![
        0.;
        (size_of::<f32>() as i64 * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic)
            as usize
    ];
    let mut U = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_in_h as usize)
            * (ti.tile_in_w as usize)
            * (us.oc as usize)
            * (us.ic as usize)
    ];
    let mut V = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_in_h as usize)
            * (ti.tile_in_w as usize)
            * (vs.num_tiles as usize)
            * (vs.ic as usize)
    ];
    let mut M = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_in_h as usize)
            * (ti.tile_in_w as usize)
            * (us.oc as usize)
            * (vs.num_tiles as usize)
    ];
    let mut Y = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_out_h as usize)
            * (ti.tile_in_w as usize)
            * (os.oc as usize)
            * (ti.num_tiles as usize)
    ];

    filter_packing(filter, &mut packed_filter, fs);
    filter_transform_w(&packed_filter, &mut U, fs, us, us.oc * us.ic);
    filter_transform_h(&mut U, fs, us, us.oc * us.ic);
    image_packing(image, &mut packed_image, is, ti);
    image_transform_w(&packed_image, &mut V, vs, ti, vs.ic * vs.num_tiles);
    image_transform_h(&mut V, vs, ti, vs.ic * vs.num_tiles);

    M.chunks_mut((us.oc * vs.num_tiles) as usize)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let h = idx as i64 / ti.tile_in_h;
            let w = idx as i64 % ti.tile_in_h;
            let A =
                &V[(h * ti.tile_in_w * vs.num_tiles * vs.ic + w * vs.num_tiles * us.ic) as usize..];
            let B = &U[(h * ti.tile_in_w * us.oc * us.ic + w * us.oc * us.ic) as usize..];
            sgemm(vs.num_tiles, us.oc, us.ic, A, B, chunk);
        });

    output_transform_w(&M, &mut Y, ti, us.oc * vs.num_tiles);
    output_transform_h(&mut Y, ti, us.oc * vs.num_tiles);
    output_unpacking_store(&Y, out, os, ti);
}
