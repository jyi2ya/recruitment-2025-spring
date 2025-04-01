#![allow(dead_code, non_snake_case, unused_assignments)]
use core::mem::size_of;
use cubecl::{
    linalg::tensor::TensorHandle,
    prelude::*,
    wgpu::{WgpuDevice, WgpuRuntime},
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator as _, ParallelIterator},
    slice::ParallelSliceMut,
};

#[inline]
fn timer(label: &str) -> scope_timer::ScopeTimer {
    scope_timer::ScopeTimer::new(label, scope_timer::TimeFormat::Milliseconds, None, false)
}

#[derive(Copy, Clone)]
struct TileIndex {
    pub batch: i64,
    pub tile_h: i64,
    pub tile_w: i64,
}
#[derive(Copy, Clone)]
struct FilterShape {
    pub output_channel: i64,
    pub input_channel: i64,
    pub h: i64,
    pub w: i64,
}
#[derive(Copy, Clone)]
struct InShape {
    pub batch_size: i64,
    pub input_channel: i64,
    pub h: i64,
    pub w: i64,
}
#[derive(Copy, Clone)]
struct UShape {
    pub h: i64,
    pub w: i64,
    pub output_channel: i64,
    pub input_channel: i64,
}
#[derive(Copy, Clone)]
struct VShape {
    pub h: i64,
    pub w: i64,
    pub num_tiles: i64,
    pub input_channel: i64,
}
#[derive(Copy, Clone)]
struct OutShape {
    pub batch_size: i64,
    pub output_channel: i64,
    pub h: i64,
    pub w: i64,
}
#[derive(Copy, Clone)]
struct TilingInfo {
    pub batch_size: i64,
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
fn get_output_shape(is: InShape, fs: FilterShape) -> OutShape {
    OutShape {
        batch_size: is.batch_size,
        output_channel: fs.output_channel,
        h: is.h - fs.h + 1,
        w: is.w - fs.w + 1,
    }
}

#[inline]
fn get_tiling_info(is: InShape, os: OutShape) -> TilingInfo {
    let tiles_on_h = (os.h + 4 - 1) / 4;
    let tiles_on_w = (os.w + 4 - 1) / 4;
    let num_tile_per_image = tiles_on_h * tiles_on_w;
    TilingInfo {
        tiles_on_h,
        tiles_on_w,
        batch_size: is.batch_size,
        num_tile_per_image,
        num_tiles: num_tile_per_image * is.batch_size,
        tile_in_h: 6,
        tile_in_w: 6,
        tile_out_h: 4,
        tile_out_w: 4,
    }
}

#[inline]
fn get_U_shape(fs: FilterShape, ti: TilingInfo) -> UShape {
    UShape {
        output_channel: fs.output_channel,
        input_channel: fs.input_channel,
        h: ti.tile_in_h,
        w: ti.tile_in_w,
    }
}

#[inline]
fn get_V_shape(is: InShape, ti: TilingInfo) -> VShape {
    VShape {
        num_tiles: ti.num_tiles,
        input_channel: is.input_channel,
        h: ti.tile_in_h,
        w: ti.tile_in_w,
    }
}

#[inline]
fn get_tile_index(tile: i64, ts: TilingInfo) -> TileIndex {
    TileIndex {
        batch: tile / ts.num_tile_per_image,
        tile_h: tile % ts.num_tile_per_image / ts.tiles_on_w,
        tile_w: tile % ts.num_tile_per_image % ts.tiles_on_w,
    }
}

fn image_transform_h(V: &mut [f32], vs: VShape, _ti: TilingInfo, collapsed_dim_size: i64) {
    V.par_chunks_exact_mut((vs.w * collapsed_dim_size) as usize)
        .for_each(|chunk| {
            for idx in 0..collapsed_dim_size {
                let load = chunk[(0 * collapsed_dim_size + idx) as usize];
                let mut z0 = 4.0f32 * load;

                let load = chunk[(1 * collapsed_dim_size + idx) as usize];
                let mut z1 = -4.0f32 * load;
                let mut z2 = 4.0f32 * load;
                let mut z3 = -2.0f32 * load;
                let mut z4 = 2.0f32 * load;
                let mut z5 = 4.0f32 * load;

                let load = chunk[(2 * collapsed_dim_size + idx) as usize];
                z0 += -5.0f32 * load;
                z1 += -4.0f32 * load;
                z2 += -4.0f32 * load;
                z3 += -load;
                z4 += -load;

                let load = chunk[(3 * collapsed_dim_size + idx) as usize];
                z1 += load;
                z2 += -load;
                z3 += 2.0f32 * load;
                z4 += -2.0f32 * load;
                z5 += -5.0f32 * load;

                let load = chunk[(4 * collapsed_dim_size + idx) as usize];
                z0 += load;
                z1 += load;
                z2 += load;
                z3 += load;
                z4 += load;

                let load = chunk[(5 * collapsed_dim_size + idx) as usize];
                z5 += load;

                chunk[(0 * collapsed_dim_size + idx) as usize] = z0;
                chunk[(1 * collapsed_dim_size + idx) as usize] = z1;
                chunk[(2 * collapsed_dim_size + idx) as usize] = z2;
                chunk[(3 * collapsed_dim_size + idx) as usize] = z3;
                chunk[(4 * collapsed_dim_size + idx) as usize] = z4;
                chunk[(5 * collapsed_dim_size + idx) as usize] = z5;
            }
        })
}

fn image_transform_w(
    packed_image: &[f32],
    V: &mut [f32],
    _vs: VShape,
    ti: TilingInfo,
    collapsed_dim_size: i64,
) {
    let chunk_size = (ti.tile_in_w * collapsed_dim_size) as usize;
    let res = V;
    let (v0, res) = res.split_at_mut(chunk_size);
    let (v1, res) = res.split_at_mut(chunk_size);
    let (v2, res) = res.split_at_mut(chunk_size);
    let (v3, res) = res.split_at_mut(chunk_size);
    let (v4, res) = res.split_at_mut(chunk_size);
    let (v5, _res) = res.split_at_mut(chunk_size);
    let packed_image = packed_image
        .chunks_exact((ti.tile_in_w * collapsed_dim_size) as usize)
        .collect::<Vec<_>>();
    (v0, v1, v2, v3, v4, v5)
        .into_par_iter()
        .enumerate()
        .for_each(|(offset, (v0, v1, v2, v3, v4, v5))| {
            let load = packed_image[0][offset];
            let mut z0 = 4.0f32 * load;

            let load = packed_image[1][offset];
            let mut z1 = -4.0f32 * load;
            let mut z2 = 4.0f32 * load;
            let mut z3 = -2.0f32 * load;
            let mut z4 = 2.0f32 * load;
            let mut z5 = 4.0f32 * load;

            let load = packed_image[2][offset];
            z0 += -5.0f32 * load;
            z1 += -4.0f32 * load;
            z2 += -4.0f32 * load;
            z3 += -load;
            z4 += -load;

            let load = packed_image[3][offset];
            z1 += load;
            z2 += -load;
            z3 += 2.0f32 * load;
            z4 += -2.0f32 * load;
            z5 += -5.0f32 * load;

            let load = packed_image[4][offset];
            z0 += load;
            z1 += load;
            z2 += load;
            z3 += load;
            z4 += load;

            let load = packed_image[5][offset];
            z5 += load;
            *v0 = z0;
            *v1 = z1;
            *v2 = z2;
            *v3 = z3;
            *v4 = z4;
            *v5 = z5;
        });
}

fn filter_transform_w(
    packed_filter: &[f32],
    U: &mut [f32],
    fs: FilterShape,
    us: UShape,
    collapsed_dim_size: i64,
) {
    let chunk_size = (us.w * collapsed_dim_size) as usize;
    let res = U;
    let (u0, res) = res.split_at_mut(chunk_size);
    let (u1, res) = res.split_at_mut(chunk_size);
    let (u2, res) = res.split_at_mut(chunk_size);
    let (u3, res) = res.split_at_mut(chunk_size);
    let (u4, res) = res.split_at_mut(chunk_size);
    let (u5, _res) = res.split_at_mut(chunk_size);

    (u0, u1, u2, u3, u4, u5)
        .into_par_iter()
        .enumerate()
        .for_each(|(offset, (u0, u1, u2, u3, u4, u5))| {
            let z6 = packed_filter[(0 * fs.w * collapsed_dim_size) as usize + offset];
            let z0 = 1.0f32 / 4.0f32 * z6;
            let mut z1 = -1.0f32 / 6.0f32 * z6;
            let mut z2 = -1.0f32 / 6.0f32 * z6;
            let mut z3 = 1.0f32 / 24.0f32 * z6;
            let mut z4 = 1.0f32 / 24.0f32 * z6;
            let z6 = packed_filter[(1 * fs.w * collapsed_dim_size) as usize + offset];
            z1 += -1.0f32 / 6.0f32 * z6;
            z2 += 1.0f32 / 6.0f32 * z6;
            z3 += 1.0f32 / 12.0f32 * z6;
            z4 += -1.0f32 / 12.0f32 * z6;
            let z6 = packed_filter[(2 * fs.w * collapsed_dim_size) as usize + offset];
            z1 += -1.0f32 / 6.0f32 * z6;
            z2 += -1.0f32 / 6.0f32 * z6;
            z3 += 1.0f32 / 6.0f32 * z6;
            z4 += 1.0f32 / 6.0f32 * z6;
            let z5 = z6;
            *u0 = z0;
            *u1 = z1;
            *u2 = z2;
            *u3 = z3;
            *u4 = z4;
            *u5 = z5;
        });
}

fn filter_transform_h(U: &mut [f32], _fs: FilterShape, us: UShape, collapsed_dim_size: i64) {
    U.par_chunks_exact_mut((us.w * collapsed_dim_size) as usize)
        .for_each(|chunk| {
            for idx in 0..collapsed_dim_size {
                let z6 = chunk[(0 * collapsed_dim_size + idx) as usize];
                let z0 = 1.0f32 / 4.0f32 * z6;
                let mut z1 = -1.0f32 / 6.0f32 * z6;
                let mut z2 = -1.0f32 / 6.0f32 * z6;
                let mut z3 = 1.0f32 / 24.0f32 * z6;
                let mut z4 = 1.0f32 / 24.0f32 * z6;
                let z6 = chunk[(1 * collapsed_dim_size + idx) as usize];
                z1 += -1.0f32 / 6.0f32 * z6;
                z2 += 1.0f32 / 6.0f32 * z6;
                z3 += 1.0f32 / 12.0f32 * z6;
                z4 += -1.0f32 / 12.0f32 * z6;
                let z6 = chunk[(2 * collapsed_dim_size + idx) as usize];
                z1 += -1.0f32 / 6.0f32 * z6;
                z2 += -1.0f32 / 6.0f32 * z6;
                z3 += 1.0f32 / 6.0f32 * z6;
                z4 += 1.0f32 / 6.0f32 * z6;
                let z5 = z6;
                chunk[(0 * collapsed_dim_size + idx) as usize] = z0;
                chunk[(1 * collapsed_dim_size + idx) as usize] = z1;
                chunk[(2 * collapsed_dim_size + idx) as usize] = z2;
                chunk[(3 * collapsed_dim_size + idx) as usize] = z3;
                chunk[(4 * collapsed_dim_size + idx) as usize] = z4;
                chunk[(5 * collapsed_dim_size + idx) as usize] = z5;
            }
        });
}

fn output_transform_w(M: &[f32], Y: &mut [f32], ti: TilingInfo, collapsed_dim_size: i64) {
    let chunk_size = (ti.tile_in_w * collapsed_dim_size) as usize;
    let res = Y;
    let (y0, res) = res.split_at_mut(chunk_size);
    let (y1, res) = res.split_at_mut(chunk_size);
    let (y2, res) = res.split_at_mut(chunk_size);
    let (y3, _res) = res.split_at_mut(chunk_size);

    (y0, y1, y2, y3)
        .into_par_iter()
        .enumerate()
        .for_each(|(offset, (y0, y1, y2, y3))| {
            let z4 = M[(0 * ti.tile_in_w * collapsed_dim_size) as usize + offset];
            let mut z0 = z4;
            let z4 = M[(1 * ti.tile_in_w * collapsed_dim_size) as usize + offset];
            z0 += z4;
            let mut z1 = z4;
            let mut z2 = z4;
            let mut z3 = z4;
            let z4 = M[(2 * ti.tile_in_w * collapsed_dim_size) as usize + offset];
            z0 += z4;
            z1 += -z4;
            z2 += z4;
            z3 += -z4;
            let z4 = M[(3 * ti.tile_in_w * collapsed_dim_size) as usize + offset];
            z0 += z4;
            z1 += 2.0f32 * z4;
            z2 += 4.0f32 * z4;
            z3 += 8.0f32 * z4;
            let z4 = M[(4 * ti.tile_in_w * collapsed_dim_size) as usize + offset];
            z0 += z4;
            z1 += -2.0f32 * z4;
            z2 += 4.0f32 * z4;
            z3 += -8.0f32 * z4;
            let z4 = M[(5 * ti.tile_in_w * collapsed_dim_size) as usize + offset];
            z3 += z4;
            *y0 = z0;
            *y1 = z1;
            *y2 = z2;
            *y3 = z3;
        });
}

fn output_transform_h(Y: &mut [f32], ti: TilingInfo, collapsed_dim_size: i64) {
    Y.par_chunks_exact_mut((collapsed_dim_size * ti.tile_in_w) as usize)
        .for_each(|chunk| {
            for idx in 0..collapsed_dim_size {
                let z4 = chunk[(0 * collapsed_dim_size + idx) as usize];
                let mut z0 = z4;
                let z4 = chunk[(1 * collapsed_dim_size + idx) as usize];
                z0 += z4;
                let mut z1 = z4;
                let mut z2 = z4;
                let mut z3 = z4;
                let z4 = chunk[(2 * collapsed_dim_size + idx) as usize];
                z0 += z4;
                z1 += -z4;
                z2 += z4;
                z3 += -z4;
                let z4 = chunk[(3 * collapsed_dim_size + idx) as usize];
                z0 += z4;
                z1 += 2.0f32 * z4;
                z2 += 4.0f32 * z4;
                z3 += 8.0f32 * z4;
                let z4 = chunk[(4 * collapsed_dim_size + idx) as usize];
                z0 += z4;
                z1 += -2.0f32 * z4;
                z2 += 4.0f32 * z4;
                z3 += -8.0f32 * z4;
                let z4 = chunk[(5 * collapsed_dim_size + idx) as usize];
                z3 += z4;
                chunk[(0 * collapsed_dim_size + idx) as usize] = z0;
                chunk[(1 * collapsed_dim_size + idx) as usize] = z1;
                chunk[(2 * collapsed_dim_size + idx) as usize] = z2;
                chunk[(3 * collapsed_dim_size + idx) as usize] = z3;
            }
        });
}

fn filter_packing(filter: &mut [f32], packed_filter: &mut [f32], fs: FilterShape) {
    for h in 0..fs.h {
        for w in 0..fs.w {
            for oc in 0..fs.output_channel {
                for ic in 0..fs.input_channel {
                    packed_filter[(h * fs.w * fs.output_channel * fs.input_channel
                        + w * fs.output_channel * fs.input_channel
                        + oc * fs.input_channel
                        + ic) as usize] = filter[(oc * fs.input_channel * fs.h * fs.w
                        + ic * fs.h * fs.w
                        + h * fs.w
                        + w) as usize];
                }
            }
        }
    }
}

fn image_packing(image: &mut [f32], packed_image: &mut [f32], is: InShape, ti: TilingInfo) {
    packed_image
        .par_chunks_exact_mut((ti.tile_in_w * ti.num_tiles * is.input_channel) as usize)
        .enumerate()
        .for_each(|(h, chunk)| {
            for w in 0..ti.tile_in_w {
                for tile in 0..ti.num_tiles {
                    for ic in 0..is.input_channel {
                        let batch = tile / ti.num_tile_per_image;
                        let h: i64 = h.try_into().unwrap();
                        let hh: i64 = tile % ti.num_tile_per_image / ti.tiles_on_w;
                        let ww: i64 = tile % ti.num_tile_per_image % ti.tiles_on_w;

                        let image_pos = if hh * 4 + h < is.h && ww * 4 + w < is.w {
                            Some(
                                (batch * is.input_channel * is.h * is.w
                                    + ic * is.h * is.w
                                    + (hh * 4 + h) * is.w
                                    + (ww * 4 + w)) as usize,
                            )
                        } else {
                            None
                        };
                        let offset =
                            w * ti.num_tiles * is.input_channel + tile * is.input_channel + ic;
                        chunk[offset as usize] = image_pos.map(|pos| image[pos]).unwrap_or(0.);
                    }
                }
            }
        });
}

fn output_unpacking_store(Y: &[f32], out: &mut [f32], os: OutShape, ti: TilingInfo) {
    for h in 0..ti.tile_out_h {
        for w in 0..ti.tile_out_w {
            for oc in 0..os.output_channel {
                for tile in 0..ti.num_tiles {
                    let tidx: TileIndex = get_tile_index(tile, ti);
                    let batch: i64 = tidx.batch;
                    let ww: i64 = tidx.tile_w;
                    let hh: i64 = tidx.tile_h;
                    if hh * 4 + h < os.h && ww * 4 + w < os.w {
                        out[(batch * os.output_channel * os.h * os.w
                            + oc * os.h * os.w
                            + (hh * 4 + h) * os.w
                            + (ww * 4 + w)) as usize] =
                            Y[(h * ti.tile_in_w * os.output_channel * ti.num_tiles
                                + w * os.output_channel * ti.num_tiles
                                + oc * ti.num_tiles
                                + tile) as usize];
                    }
                }
            }
        }
    }
}

fn sgemm<RT, F>(device: &RT::Device, M: i64, N: i64, K: i64, A: &[F], B: &[F], C: &mut [F])
where
    F: Float + cubecl::CubeElement + cubecl::linalg::matmul::components::MatmulPrecision,
    RT: cubecl::Runtime,
{
    let client = RT::client(device);
    let M: usize = M.try_into().unwrap();
    let N: usize = N.try_into().unwrap();
    let K: usize = K.try_into().unwrap();

    let (tensor_A, tensor_B, tensor_C): (
        TensorHandle<RT, F>,
        TensorHandle<RT, F>,
        TensorHandle<RT, F>,
    ) = {
        let input_A = client.create(F::as_bytes(A));
        let input_B = client.create(F::as_bytes(B));
        let output_C = client.empty(C.len() * core::mem::size_of::<F>());

        (
            cubecl::linalg::tensor::TensorHandle::new_contiguous(vec![M, K], input_A),
            cubecl::linalg::tensor::TensorHandle::new(input_B, vec![K, N], vec![1, K]),
            cubecl::linalg::tensor::TensorHandle::new_contiguous(vec![M, N], output_C),
        )
    };

    cubecl::linalg::matmul::launch_ref::<RT, F>(
        &cubecl::linalg::matmul::Strategy::Auto,
        &client,
        &tensor_A.as_ref(),
        &tensor_B.as_ref(),
        &tensor_C.as_ref(),
    )
    .unwrap();

    let reshaped: TensorHandle<RT, F> =
        cubecl::linalg::tensor::TensorHandle::new(tensor_C.handle, vec![N, M], vec![1, N]);

    let output_C: TensorHandle<RT, F> =
        cubecl::linalg::tensor::into_contiguous(&client, &reshaped.as_ref());
    let bytes = client.read_one(output_C.handle.binding());
    let result = F::from_bytes(&bytes);

    C.copy_from_slice(result);
}

fn sgemm_cpu(M: i64, _N: i64, K: i64, A: &[f32], B: &[f32], C: &mut [f32]) {
    C.par_chunks_exact_mut(M as usize)
        .enumerate()
        .for_each(|(n, chunk)| {
            for m in 0..M {
                let mut c = 0.;
                for k in 0..K {
                    c += A[(m * K + k) as usize] * B[(n as i64 * K + k) as usize];
                }
                chunk[m as usize] = c;
            }
        });
}

fn do_test() {
    let a = vec![1., 2., 3., 4.];
    let b = [5., 6., 7., 8., 9., 10.];
    let mut c = vec![0.; 6];
    sgemm_cpu(2, 3, 2, &a, &b, &mut c);
    dbg!(c);
    let mut c = vec![0.; 6];
    sgemm::<WgpuRuntime, f32>(&WgpuDevice::default(), 2, 3, 2, &a, &b, &mut c);
    dbg!(c);
    panic!("");
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
    // do_test();

    let image_shape = InShape {
        batch_size: batch_num as i64,
        input_channel: input_channel_num as i64,
        h: image_height as i64,
        w: image_width as i64,
    };
    let image = unsafe {
        std::slice::from_raw_parts_mut(
            image,
            (image_shape.batch_size * image_shape.h * image_shape.w * image_shape.input_channel)
                as usize,
        )
    };

    let filter_shape = FilterShape {
        output_channel: output_channel_num as i64,
        input_channel: input_channel_num as i64,
        h: 3 as i64,
        w: 3 as i64,
    };
    let filter = unsafe {
        std::slice::from_raw_parts_mut(
            filter,
            (filter_shape.input_channel
                * filter_shape.h
                * filter_shape.w
                * filter_shape.output_channel) as usize,
        )
    };

    let os: OutShape = get_output_shape(image_shape, filter_shape);
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            out,
            (os.batch_size * os.h * os.w * os.output_channel) as usize,
        )
    };

    let ti: TilingInfo = get_tiling_info(image_shape, os);
    let us: UShape = get_U_shape(filter_shape, ti);
    let vs: VShape = get_V_shape(image_shape, ti);
    let mut packed_filter = vec![
        0.;
        (size_of::<f32>() as i64
            * filter_shape.h
            * filter_shape.w
            * filter_shape.output_channel
            * filter_shape.input_channel) as usize
    ];
    let mut packed_image = vec![
        0.;
        (size_of::<f32>() as i64
            * ti.tile_in_h
            * ti.tile_in_w
            * ti.num_tiles
            * image_shape.input_channel) as usize
    ];
    let mut U = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_in_h as usize)
            * (ti.tile_in_w as usize)
            * (us.output_channel as usize)
            * (us.input_channel as usize)
    ];
    let mut V = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_in_h as usize)
            * (ti.tile_in_w as usize)
            * (vs.num_tiles as usize)
            * (vs.input_channel as usize)
    ];
    let mut M = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_in_h as usize)
            * (ti.tile_in_w as usize)
            * (us.output_channel as usize)
            * (vs.num_tiles as usize)
    ];
    let mut Y = vec![
        0.;
        (::core::mem::size_of::<f32>() as usize)
            * (ti.tile_out_h as usize)
            * (ti.tile_in_w as usize)
            * (os.output_channel as usize)
            * (ti.num_tiles as usize)
    ];

    {
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _t = timer("filter preprocess");
                filter_packing(filter, &mut packed_filter, filter_shape);
                filter_transform_w(
                    &packed_filter,
                    &mut U,
                    filter_shape,
                    us,
                    us.output_channel * us.input_channel,
                );
                filter_transform_h(
                    &mut U,
                    filter_shape,
                    us,
                    us.output_channel * us.input_channel,
                );
            });
            scope.spawn(|| {
                {
                    let _t = timer("image packing");
                    image_packing(image, &mut packed_image, image_shape, ti);
                }
                let _t = timer("other image preprocess");
                image_transform_w(
                    &packed_image,
                    &mut V,
                    vs,
                    ti,
                    vs.input_channel * vs.num_tiles,
                );
                image_transform_h(&mut V, vs, ti, vs.input_channel * vs.num_tiles);
            });
        })
    }

    {
        let _t = timer("gemm       ");
        std::thread::scope(|scope| {
            let max_gpu_concurrency = 2;
            let max_cpu_concurrency = 4;

            let indexes = (0..ti.tile_in_w)
                .flat_map(|h| (0..ti.tile_in_h).map(move |w| (h, w)))
                .collect::<Vec<_>>();
            let (tx, rx) = crossbeam::channel::unbounded();

            for _ in 0..max_gpu_concurrency {
                let rx = rx.clone();
                scope.spawn(move || {
                    let device = WgpuDevice::default();
                    rx.into_iter().for_each(|(M, N, K, A, B, C)| {
                        sgemm::<WgpuRuntime, _>(&device, M, N, K, A, B, C);
                    });
                });
            }

            for _ in 0..max_cpu_concurrency {
                let rx = rx.clone();
                scope.spawn(move || {
                    rx.into_iter().for_each(|(M, N, K, A, B, C)| {
                        sgemm_cpu(M, N, K, A, B, C);
                    });
                });
            }

            M.chunks_mut((us.output_channel * vs.num_tiles) as usize)
                .zip(indexes)
                .for_each(|(chunk, (h, w))| {
                    let a_begin = (h * ti.tile_in_w * vs.num_tiles * vs.input_channel
                        + w * vs.num_tiles * us.input_channel)
                        as usize;
                    let a_end = a_begin + (vs.num_tiles * us.input_channel) as usize;
                    let A = &V[a_begin..a_end];
                    let b_begin = (h * ti.tile_in_w * us.output_channel * us.input_channel
                        + w * us.output_channel * us.input_channel)
                        as usize;
                    let b_end = b_begin + (us.output_channel * us.input_channel) as usize;
                    let B = &U[b_begin..b_end];
                    tx.send((
                        vs.num_tiles,
                        us.output_channel,
                        us.input_channel,
                        A,
                        B,
                        chunk,
                    ))
                    .unwrap();
                });
        });
    }

    {
        let _t = timer("postprocess");
        output_transform_w(&M, &mut Y, ti, us.output_channel * vs.num_tiles);
        output_transform_h(&mut Y, ti, us.output_channel * vs.num_tiles);
        output_unpacking_store(&Y, out, os, ti);
    }
}
