#![allow(
    non_snake_case,
    unused_assignments,
    clippy::erasing_op,
    clippy::identity_op
)]
use cubecl::{linalg::tensor::TensorHandle, prelude::*};

#[inline]
fn timer(label: &str) -> scope_timer::ScopeTimer {
    scope_timer::ScopeTimer::new(label, scope_timer::TimeFormat::Milliseconds, None, false)
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct FilterShape {
    pub output_channel: u32,
    pub input_channel: u32,
    pub h: u32,
    pub w: u32,
}
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct InShape {
    pub batch_size: u32,
    pub input_channel: u32,
    pub h: u32,
    pub w: u32,
}
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct UShape {
    pub h: u32,
    pub w: u32,
    pub output_channel: u32,
    pub input_channel: u32,
}
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct VShape {
    pub h: u32,
    pub w: u32,
    pub num_tiles: u32,
    pub input_channel: u32,
}
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct OutShape {
    pub batch_size: u32,
    pub output_channel: u32,
    pub h: u32,
    pub w: u32,
}
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct TilingInfo {
    pub batch_size: u32,
    pub num_tile_per_image: u32,
    pub num_tiles: u32,
    pub tiles_on_h: u32,
    pub tiles_on_w: u32,
    pub tile_in_h: u32,
    pub tile_in_w: u32,
    pub tile_out_h: u32,
    pub tile_out_w: u32,
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
    let tiles_on_h = os.h.div_ceil(4);
    let tiles_on_w = os.w.div_ceil(4);
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

#[cube(launch)]
fn image_transform_w_gpu(
    packed_image: &Tensor<f32>,
    V: &mut Tensor<f32>,
    #[comptime] vs: VShape,
    #[comptime] ti: TilingInfo,
) {
    let w = ABSOLUTE_POS_X;
    let ic = ABSOLUTE_POS_Y;
    let tile = ABSOLUTE_POS_Z;

    if w < ti.tile_in_w && ic < vs.input_channel && tile < vs.num_tiles {
        let idx = ic * vs.num_tiles + tile;
        let collapsed_dim_size = vs.input_channel * vs.num_tiles;

        let offset = collapsed_dim_size * w + idx;

        let load = packed_image[0 * ti.tile_in_w * collapsed_dim_size + offset];
        let mut z0 = 4.0f32 * load;

        let load = packed_image[1 * ti.tile_in_w * collapsed_dim_size + offset];
        let mut z1 = -4.0f32 * load;
        let mut z2 = 4.0f32 * load;
        let mut z3 = -2.0f32 * load;
        let mut z4 = 2.0f32 * load;
        let mut z5 = 4.0f32 * load;

        let load = packed_image[2 * ti.tile_in_w * collapsed_dim_size + offset];
        z0 += -5.0f32 * load;
        z1 += -4.0f32 * load;
        z2 += -4.0f32 * load;
        z3 += -load;
        z4 += -load;

        let load = packed_image[3 * ti.tile_in_w * collapsed_dim_size + offset];
        z1 += load;
        z2 += -load;
        z3 += 2.0f32 * load;
        z4 += -2.0f32 * load;
        z5 += -5.0f32 * load;

        let load = packed_image[4 * ti.tile_in_w * collapsed_dim_size + offset];
        z0 += load;
        z1 += load;
        z2 += load;
        z3 += load;
        z4 += load;

        let load = packed_image[5 * ti.tile_in_w * collapsed_dim_size + offset];
        z5 += load;

        V[0 * ti.tile_in_w * collapsed_dim_size + offset] = z0;
        V[1 * ti.tile_in_w * collapsed_dim_size + offset] = z1;
        V[2 * ti.tile_in_w * collapsed_dim_size + offset] = z2;
        V[3 * ti.tile_in_w * collapsed_dim_size + offset] = z3;
        V[4 * ti.tile_in_w * collapsed_dim_size + offset] = z4;
        V[5 * ti.tile_in_w * collapsed_dim_size + offset] = z5;
    }
}

#[cube(launch)]
fn image_transform_h_gpu(V: &mut Tensor<f32>, #[comptime] vs: VShape, #[comptime] ti: TilingInfo) {
    let h = ABSOLUTE_POS_X;
    let ic = ABSOLUTE_POS_Y;
    let tile = ABSOLUTE_POS_Z;

    if h < ti.tile_in_h && ic < vs.input_channel && tile < vs.num_tiles {
        let idx = ic * vs.num_tiles + tile;
        let collapsed_dim_size = vs.input_channel * vs.num_tiles;

        let load = V[h * vs.w * collapsed_dim_size + 0 * collapsed_dim_size + idx];
        let mut z0 = 4.0f32 * load;

        let load = V[h * vs.w * collapsed_dim_size + 1 * collapsed_dim_size + idx];
        let mut z1 = -4.0f32 * load;
        let mut z2 = 4.0f32 * load;
        let mut z3 = -2.0f32 * load;
        let mut z4 = 2.0f32 * load;
        let mut z5 = 4.0f32 * load;

        let load = V[h * vs.w * collapsed_dim_size + 2 * collapsed_dim_size + idx];
        z0 += -5.0f32 * load;
        z1 += -4.0f32 * load;
        z2 += -4.0f32 * load;
        z3 += -load;
        z4 += -load;

        let load = V[h * vs.w * collapsed_dim_size + 3 * collapsed_dim_size + idx];
        z1 += load;
        z2 += -load;
        z3 += 2.0f32 * load;
        z4 += -2.0f32 * load;
        z5 += -5.0f32 * load;

        let load = V[h * vs.w * collapsed_dim_size + 4 * collapsed_dim_size + idx];
        z0 += load;
        z1 += load;
        z2 += load;
        z3 += load;
        z4 += load;

        let load = V[h * vs.w * collapsed_dim_size + 5 * collapsed_dim_size + idx];
        z5 += load;

        V[h * vs.w * collapsed_dim_size + 0 * collapsed_dim_size + idx] = z0;
        V[h * vs.w * collapsed_dim_size + 1 * collapsed_dim_size + idx] = z1;
        V[h * vs.w * collapsed_dim_size + 2 * collapsed_dim_size + idx] = z2;
        V[h * vs.w * collapsed_dim_size + 3 * collapsed_dim_size + idx] = z3;
        V[h * vs.w * collapsed_dim_size + 4 * collapsed_dim_size + idx] = z4;
        V[h * vs.w * collapsed_dim_size + 5 * collapsed_dim_size + idx] = z5;
    }
}

#[cube(launch)]
fn filter_transform_w_gpu(
    packed_filter: &Tensor<f32>,
    U: &mut Tensor<f32>,
    #[comptime] fs: FilterShape,
    #[comptime] us: UShape,
) {
    let w = ABSOLUTE_POS_X;
    let oc = ABSOLUTE_POS_Y;
    let ic = ABSOLUTE_POS_Z;

    if w < fs.w && oc < us.output_channel && ic < us.input_channel {
        let collapsed_dim_size = us.input_channel * us.output_channel;
        let idx = oc * us.input_channel + ic;

        let z6 = packed_filter[0 * fs.w * collapsed_dim_size + w * collapsed_dim_size + idx];
        let z0 = 1.0f32 / 4.0f32 * z6;
        let mut z1 = -1.0f32 / 6.0f32 * z6;
        let mut z2 = -1.0f32 / 6.0f32 * z6;
        let mut z3 = 1.0f32 / 24.0f32 * z6;
        let mut z4 = 1.0f32 / 24.0f32 * z6;
        let z6 = packed_filter[1 * fs.w * collapsed_dim_size + w * collapsed_dim_size + idx];
        z1 += -1.0f32 / 6.0f32 * z6;
        z2 += 1.0f32 / 6.0f32 * z6;
        z3 += 1.0f32 / 12.0f32 * z6;
        z4 += -1.0f32 / 12.0f32 * z6;
        let z6 = packed_filter[2 * fs.w * collapsed_dim_size + w * collapsed_dim_size + idx];
        z1 += -1.0f32 / 6.0f32 * z6;
        z2 += -1.0f32 / 6.0f32 * z6;
        z3 += 1.0f32 / 6.0f32 * z6;
        z4 += 1.0f32 / 6.0f32 * z6;
        let z5 = z6;
        U[0 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx] = z0;
        U[1 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx] = z1;
        U[2 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx] = z2;
        U[3 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx] = z3;
        U[4 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx] = z4;
        U[5 * us.w * collapsed_dim_size + w * collapsed_dim_size + idx] = z5;
    }
}

#[cube(launch)]
fn filter_transform_h_gpu(U: &mut Tensor<f32>, #[comptime] us: UShape) {
    let h = ABSOLUTE_POS_X;
    let oc = ABSOLUTE_POS_Y;
    let ic = ABSOLUTE_POS_Z;

    if h < us.h && oc < us.output_channel && ic < us.input_channel {
        let collapsed_dim_size = us.input_channel * us.output_channel;
        let idx = oc * us.input_channel + ic;

        let z6 = U[h * us.w * collapsed_dim_size + 0 * collapsed_dim_size + idx];
        let z0 = 1.0f32 / 4.0f32 * z6;
        let mut z1 = -1.0f32 / 6.0f32 * z6;
        let mut z2 = -1.0f32 / 6.0f32 * z6;
        let mut z3 = 1.0f32 / 24.0f32 * z6;
        let mut z4 = 1.0f32 / 24.0f32 * z6;
        let z6 = U[h * us.w * collapsed_dim_size + 1 * collapsed_dim_size + idx];
        z1 += -1.0f32 / 6.0f32 * z6;
        z2 += 1.0f32 / 6.0f32 * z6;
        z3 += 1.0f32 / 12.0f32 * z6;
        z4 += -1.0f32 / 12.0f32 * z6;
        let z6 = U[h * us.w * collapsed_dim_size + 2 * collapsed_dim_size + idx];
        z1 += -1.0f32 / 6.0f32 * z6;
        z2 += -1.0f32 / 6.0f32 * z6;
        z3 += 1.0f32 / 6.0f32 * z6;
        z4 += 1.0f32 / 6.0f32 * z6;
        let z5 = z6;
        U[h * us.w * collapsed_dim_size + 0 * collapsed_dim_size + idx] = z0;
        U[h * us.w * collapsed_dim_size + 1 * collapsed_dim_size + idx] = z1;
        U[h * us.w * collapsed_dim_size + 2 * collapsed_dim_size + idx] = z2;
        U[h * us.w * collapsed_dim_size + 3 * collapsed_dim_size + idx] = z3;
        U[h * us.w * collapsed_dim_size + 4 * collapsed_dim_size + idx] = z4;
        U[h * us.w * collapsed_dim_size + 5 * collapsed_dim_size + idx] = z5;
    }
}

#[cube(launch)]
fn filter_packing_gpu(
    filter: &Tensor<Line<f32>>,
    packed_filter: &mut Tensor<Line<f32>>,
    #[comptime] fs: FilterShape,
) {
    let h = ABSOLUTE_POS_X;
    let w = ABSOLUTE_POS_Y;
    let oc = ABSOLUTE_POS_Z;
    if h < fs.h && w < fs.w && oc < fs.output_channel {
        #[unroll]
        for ic in 0..fs.input_channel {
            let packed_idx = h * fs.w * fs.output_channel * fs.input_channel
                + w * fs.output_channel * fs.input_channel
                + oc * fs.input_channel
                + ic;
            let filter_idx = oc * fs.input_channel * fs.h * fs.w + ic * fs.h * fs.w + h * fs.w + w;
            packed_filter[packed_idx] = filter[filter_idx];
        }
    }
}

#[cube(launch)]
fn image_packing_gpu(
    image: &Tensor<f32>,
    packed_image: &mut Tensor<f32>,
    #[comptime] is: InShape,
    #[comptime] ti: TilingInfo,
) {
    let h = ABSOLUTE_POS_X;
    let w = ABSOLUTE_POS_Y;
    let batch = ABSOLUTE_POS_Z;

    if h < ti.tile_in_h && w < ti.tile_in_w && batch < ti.batch_size && is.h > h && is.w > w {
        let max_hh = (is.h - h + 3) / 4;
        let max_ww = (is.w - w + 3) / 4;

        let hb = if max_hh < ti.tiles_on_h {
            max_hh
        } else {
            ti.tiles_on_h
        };

        let wb = if max_ww < ti.tiles_on_w {
            max_ww
        } else {
            ti.tiles_on_w
        };

        let base =
            h * ti.tile_in_w * ti.batch_size * ti.tiles_on_h * ti.tiles_on_w * is.input_channel
                + w * ti.batch_size * ti.tiles_on_h * ti.tiles_on_w * is.input_channel
                + batch * ti.tiles_on_h * ti.tiles_on_w * is.input_channel;
        for hh in 0..hb {
            let base = base + hh * ti.tiles_on_w * is.input_channel;
            for ww in 0..wb {
                let base = base + ww * is.input_channel;

                #[unroll]
                for ic in 0..is.input_channel {
                    let offset = base + ic;
                    let image_pos = batch * is.input_channel * is.h * is.w
                        + ic * is.h * is.w
                        + (hh * 4 + h) * is.w
                        + (ww * 4 + w);
                    packed_image[offset] = image[image_pos];
                }
            }
        }
    }
}

#[cube]
fn get_index(
    h: u32,
    w: u32,
    oc: u32,
    tile: u32,
    #[comptime] ti: TilingInfo,
    #[comptime] us: UShape,
    #[comptime] vs: VShape,
) -> u32 {
    let dims = (
        (h, ti.tile_in_h),
        (oc, us.output_channel),
        (tile, vs.num_tiles),
        (w, ti.tile_in_w),
    );
    ((dims.0 .0 * dims.1 .1 + dims.1 .0) * dims.2 .1 + dims.2 .0) * dims.3 .1 + dims.3 .0
}

#[cube(launch)]
fn output_transform_w_gpu(
    M: &Tensor<f32>,
    Y: &mut Tensor<f32>,
    #[comptime] ti: TilingInfo,
    #[comptime] us: UShape,
    #[comptime] vs: VShape,
) {
    let w = ABSOLUTE_POS_X;
    let oc = ABSOLUTE_POS_Y;
    let tile = ABSOLUTE_POS_Z;
    if w < ti.tile_in_w && oc < us.output_channel && tile < vs.num_tiles {
        let collapsed_dim_size = us.output_channel * vs.num_tiles;
        let idx = oc * vs.num_tiles + tile;

        let base = w * collapsed_dim_size + idx;

        let m = (
            M[0 * ti.tile_in_w * collapsed_dim_size + base],
            M[1 * ti.tile_in_w * collapsed_dim_size + base],
            M[2 * ti.tile_in_w * collapsed_dim_size + base],
            M[3 * ti.tile_in_w * collapsed_dim_size + base],
            M[4 * ti.tile_in_w * collapsed_dim_size + base],
            M[5 * ti.tile_in_w * collapsed_dim_size + base],
        );

        let a = (1., 1., 1., 1., 1., 0.);
        let val = m.0 * a.0 + m.1 * a.1 + m.2 * a.2 + m.3 * a.3 + m.4 * a.4 + m.5 * a.5;
        Y[get_index(0, w, oc, tile, ti, us, vs)] = val;
        let a = (0., 1., -1., 2.0f32, -2.0f32, 0.);
        let val = m.0 * a.0 + m.1 * a.1 + m.2 * a.2 + m.3 * a.3 + m.4 * a.4 + m.5 * a.5;
        Y[get_index(1, w, oc, tile, ti, us, vs)] = val;
        let a = (0., 1., 1., 4.0f32, 4.0f32, 0.);
        let val = m.0 * a.0 + m.1 * a.1 + m.2 * a.2 + m.3 * a.3 + m.4 * a.4 + m.5 * a.5;
        Y[get_index(2, w, oc, tile, ti, us, vs)] = val;
        let a = (0., 1., -1., 8.0f32, -8.0f32, 1.);
        let val = m.0 * a.0 + m.1 * a.1 + m.2 * a.2 + m.3 * a.3 + m.4 * a.4 + m.5 * a.5;
        Y[get_index(3, w, oc, tile, ti, us, vs)] = val;
    }
}

#[cube(launch)]
fn output_transform_h_gpu(
    Y: &mut Tensor<f32>,
    #[comptime] ti: TilingInfo,
    #[comptime] us: UShape,
    #[comptime] vs: VShape,
) {
    let h = ABSOLUTE_POS_X;
    let oc = ABSOLUTE_POS_Y;
    let tile = ABSOLUTE_POS_Z;

    if h < ti.tile_in_h && oc < us.output_channel && tile < vs.num_tiles {
        let z4 = Y[get_index(h, 0, oc, tile, ti, us, vs)];
        let mut z0 = z4;
        let z4 = Y[get_index(h, 1, oc, tile, ti, us, vs)];
        z0 += z4;
        let mut z1 = z4;
        let mut z2 = z4;
        let mut z3 = z4;
        let z4 = Y[get_index(h, 2, oc, tile, ti, us, vs)];
        z0 += z4;
        z1 += -z4;
        z2 += z4;
        z3 += -z4;
        let z4 = Y[get_index(h, 3, oc, tile, ti, us, vs)];
        z0 += z4;
        z1 += 2.0f32 * z4;
        z2 += 4.0f32 * z4;
        z3 += 8.0f32 * z4;
        let z4 = Y[get_index(h, 4, oc, tile, ti, us, vs)];
        z0 += z4;
        z1 += -2.0f32 * z4;
        z2 += 4.0f32 * z4;
        z3 += -8.0f32 * z4;
        let z4 = Y[get_index(h, 5, oc, tile, ti, us, vs)];
        z3 += z4;
        Y[get_index(h, 0, oc, tile, ti, us, vs)] = z0;
        Y[get_index(h, 1, oc, tile, ti, us, vs)] = z1;
        Y[get_index(h, 2, oc, tile, ti, us, vs)] = z2;
        Y[get_index(h, 3, oc, tile, ti, us, vs)] = z3;
    }
}

#[cube(launch)]
fn output_unpacking_store_gpu(
    Y: &Tensor<f32>,
    out: &mut Tensor<f32>,
    #[comptime] os: OutShape,
    #[comptime] ti: TilingInfo,
    #[comptime] us: UShape,
    #[comptime] vs: VShape,
) {
    let batch = ABSOLUTE_POS_X;
    let oc = ABSOLUTE_POS_Y;
    if batch < os.batch_size && oc < os.output_channel {
        let base = (oc + os.output_channel * batch) * (os.h * os.w);
        for gh in 0..os.h {
            for gw in 0..os.w {
                let h = gh % 4;
                let w = gw % 4;
                let hh = gh / 4;
                let ww = gw / 4;
                let tile = batch * ti.num_tile_per_image + hh * ti.tiles_on_w + ww;

                out[base + gh * os.w + gw] = Y[get_index(h, w, oc, tile, ti, us, vs)];
            }
        }
    }
}

#[cube(launch)]
fn wow_sgemm_gpu(
    M: &mut Tensor<f32>,
    U: &Tensor<f32>,
    V: &Tensor<f32>,
    #[comptime] ti: TilingInfo,
    #[comptime] us: UShape,
    #[comptime] vs: VShape,
) {
    let tile = ABSOLUTE_POS_X;
    let n = ABSOLUTE_POS_Y;
    let m = ABSOLUTE_POS_Z;
    if tile < ti.tile_in_h * ti.tile_in_w && n < us.output_channel && m < vs.num_tiles {
        let offset = tile * us.output_channel * vs.num_tiles + n * vs.num_tiles + m;
        let a_base = tile * vs.num_tiles * us.input_channel + m * us.input_channel;
        let b_base = tile * us.output_channel * us.input_channel + n * us.input_channel;

        let mut sum = 0.;

        #[unroll]
        for k in 0..us.input_channel {
            sum += V[a_base + k] * U[b_base + k];
        }

        M[offset] = sum;
    }
}

fn create_tensor<RT: cubecl::Runtime>(
    client: &ComputeClient<RT::Server, RT::Channel>,
    shape: Vec<u32>,
    mem: &[f32],
) -> TensorHandle<RT, f32> {
    let shape = shape
        .into_iter()
        .map(|size| usize::try_from(size).unwrap())
        .collect::<Vec<_>>();
    let handle = client.create(f32::as_bytes(mem));
    TensorHandle::new_contiguous(shape, handle)
}

fn alloc_tensor<RT: cubecl::Runtime>(
    client: &ComputeClient<RT::Server, RT::Channel>,
    shape: Vec<u32>,
) -> TensorHandle<RT, f32> {
    let shape = shape
        .into_iter()
        .map(|size| usize::try_from(size).unwrap())
        .collect::<Vec<_>>();
    let len = shape.iter().product::<usize>();
    let handle = client.empty(core::mem::size_of::<f32>() * len);
    TensorHandle::new_contiguous(shape, handle)
}

fn get_cube_count(shape: [u32; 3], tiling: CubeDim) -> CubeCount {
    CubeCount::new_3d(
        shape[0].div_ceil(tiling.x),
        shape[1].div_ceil(tiling.y),
        shape[2].div_ceil(tiling.z),
    )
}

async fn winograd_convolution_rs<RT>(
    image_shape: InShape,
    image: &[f32],
    filter_shape: FilterShape,
    filter: &[f32],
    os: OutShape,
    out: &mut [f32],
) where
    RT: cubecl::Runtime,
{
    let ti: TilingInfo = get_tiling_info(image_shape, os);
    let us: UShape = get_U_shape(filter_shape, ti);
    let vs: VShape = get_V_shape(image_shape, ti);
    let device = RT::Device::default();
    let client = RT::client(&device);

    // dbg!(client.memory_usage());

    let U = {
        let packed_filter = {
            let filter = create_tensor::<RT>(
                &client,
                vec![
                    filter_shape.h,
                    filter_shape.w,
                    filter_shape.output_channel,
                    filter_shape.input_channel,
                ],
                filter,
            );

            let packed_filter = alloc_tensor::<RT>(
                &client,
                vec![
                    filter_shape.h,
                    filter_shape.w,
                    filter_shape.output_channel,
                    filter_shape.input_channel,
                ],
            );
            let cube_dim = CubeDim::new_3d(1, 16, 16);
            let cube_count = get_cube_count(
                [filter_shape.h, filter_shape.w, filter_shape.output_channel],
                cube_dim,
            );
            // dbg!(&cube_count);
            filter_packing_gpu::launch(
                &client,
                cube_count,
                cube_dim,
                filter.as_arg(1),
                packed_filter.as_arg(1),
                filter_shape,
            );

            packed_filter
        };

        let U_tensor = alloc_tensor(
            &client,
            vec![us.h, us.w, us.output_channel, us.input_channel],
        );

        let cube_dim = CubeDim::new_3d(1, 16, 16);
        let cube_count = get_cube_count(
            [filter_shape.w, us.output_channel, us.input_channel],
            cube_dim,
        );
        // dbg!(&cube_count);
        filter_transform_w_gpu::launch(
            &client,
            cube_count,
            cube_dim,
            packed_filter.as_arg(1),
            U_tensor.as_arg(1),
            filter_shape,
            us,
        );

        let cube_dim = CubeDim::new_3d(1, 16, 16);
        let cube_count = get_cube_count([us.h, us.output_channel, us.input_channel], cube_dim);
        // dbg!(&cube_count);
        filter_transform_h_gpu::launch(&client, cube_count, cube_dim, U_tensor.as_arg(1), us);

        // dbg!(client.memory_usage());
        U_tensor
    };

    // dbg!(client.memory_usage());

    let V = {
        let packed_image = {
            let image_tensor = create_tensor::<RT>(
                &client,
                vec![
                    ti.tile_in_h,
                    ti.tile_in_w,
                    ti.num_tiles,
                    image_shape.input_channel,
                ],
                image,
            );
            let packed = alloc_tensor(
                &client,
                vec![
                    ti.tile_in_h,
                    ti.tile_in_w,
                    ti.num_tiles,
                    image_shape.input_channel,
                ],
            );
            let cube_dim = CubeDim::new_3d(1, 16, 16);
            let cube_count = get_cube_count([ti.tile_in_h, ti.tile_in_w, ti.batch_size], cube_dim);
            // dbg!(&cube_count);
            image_packing_gpu::launch(
                &client,
                cube_count,
                cube_dim,
                image_tensor.as_arg(1),
                packed.as_arg(1),
                image_shape,
                ti,
            );

            packed
        };

        let V_tensor = alloc_tensor(
            &client,
            vec![ti.tile_in_h, ti.tile_in_w, vs.num_tiles, vs.input_channel],
        );

        let cube_dim = CubeDim::new_3d(1, 16, 16);
        let cube_count = get_cube_count([ti.tile_in_w, vs.input_channel, vs.num_tiles], cube_dim);
        // dbg!(&cube_count);
        image_transform_w_gpu::launch(
            &client,
            cube_count,
            cube_dim,
            packed_image.as_arg(1),
            V_tensor.as_arg(1),
            vs,
            ti,
        );

        let cube_dim = CubeDim::new_3d(1, 16, 16);
        let cube_count = get_cube_count([ti.tile_in_h, vs.input_channel, vs.num_tiles], cube_dim);
        // dbg!(&cube_count);
        image_transform_h_gpu::launch(&client, cube_count, cube_dim, V_tensor.as_arg(1), vs, ti);

        // dbg!(client.memory_usage());
        V_tensor
    };

    client.sync().await;

    let M = {
        let _t = timer("wow gemm");
        let M_tensor = alloc_tensor(
            &client,
            vec![ti.tile_in_h, ti.tile_in_w, us.output_channel, vs.num_tiles],
        );

        let cube_dim = CubeDim::new_3d(4, 8, 8);
        let cube_count = get_cube_count(
            [ti.tile_in_h * ti.tile_in_w, us.output_channel, vs.num_tiles],
            cube_dim,
        );
        // dbg!(&cube_count);
        wow_sgemm_gpu::launch(
            &client,
            cube_count,
            cube_dim,
            M_tensor.as_arg(1),
            U.as_arg(1),
            V.as_arg(1),
            ti,
            us,
            vs,
        );

        drop(U);
        drop(V);

        client.sync().await;
        // dbg!(client.memory_usage());
        M_tensor
    };

    // dbg!(client.memory_usage());

    let result = {
        let _t = timer("unpack output");

        let Y_tensor = {
            let Y_tensor = alloc_tensor::<RT>(
                &client,
                vec![ti.tile_out_h, ti.tile_in_w, os.output_channel, ti.num_tiles],
            );

            let cube_dim = CubeDim::new_3d(1, 16, 16);
            let cube_count =
                get_cube_count([ti.tile_in_w, us.output_channel, vs.num_tiles], cube_dim);
            // dbg!(&cube_count);
            {
                let _t = timer("transform_w");
                output_transform_w_gpu::launch(
                    &client,
                    cube_count,
                    cube_dim,
                    M.as_arg(1),
                    Y_tensor.as_arg(1),
                    ti,
                    us,
                    vs,
                );
                client.sync().await;
            }

            drop(M);
            client.sync().await;
            Y_tensor
        };

        let cube_dim = CubeDim::new_3d(1, 16, 16);
        let cube_count = get_cube_count([ti.tile_out_h, us.output_channel, vs.num_tiles], cube_dim);
        // dbg!(&cube_count);
        {
            let _t = timer("transform h");
            output_transform_h_gpu::launch(
                &client,
                cube_count,
                cube_dim,
                Y_tensor.as_arg(1),
                ti,
                us,
                vs,
            );
            client.sync().await;
        }

        let result =
            alloc_tensor::<RT>(&client, vec![os.batch_size, os.output_channel, os.h, os.w]);
        let cube_dim = CubeDim::new_2d(16, 16);
        let cube_count = get_cube_count([os.batch_size, os.output_channel, 1], cube_dim);
        // dbg!(&cube_count);
        {
            let _t = timer("real unpack");

            output_unpacking_store_gpu::launch(
                &client,
                cube_count,
                cube_dim,
                Y_tensor.as_arg(1),
                result.as_arg(1),
                os,
                ti,
                us,
                vs,
            );
            client.sync().await;
        }
        // dbg!(client.memory_usage());
        result
    };

    // dbg!(client.memory_usage());

    let data = {
        let _t = timer("memcpy device to host");
        client.read_one_async(result.handle.binding()).await
    };
    let _t = timer("memcpy");
    out.copy_from_slice(f32::from_bytes(&data));
}

/// # Safety
///
/// 我觉得这很 safe
#[unsafe(no_mangle)]
pub unsafe extern "C" fn winograd_convolution(
    image: *mut libc::c_float,
    image_height: libc::c_int,
    image_width: libc::c_int,
    input_channel_num: libc::c_int,
    filter: *mut libc::c_float,
    output_channel_num: libc::c_int,
    batch_num: libc::c_int,
    out: *mut libc::c_float,
) {
    let image_shape = InShape {
        batch_size: batch_num as u32,
        input_channel: input_channel_num as u32,
        h: image_height as u32,
        w: image_width as u32,
    };

    let image = unsafe {
        std::slice::from_raw_parts_mut(
            image,
            (image_shape.batch_size * image_shape.h * image_shape.w * image_shape.input_channel)
                as usize,
        )
    };

    let filter_shape = FilterShape {
        output_channel: output_channel_num as u32,
        input_channel: input_channel_num as u32,
        h: 3_u32,
        w: 3_u32,
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

    pollster::block_on(winograd_convolution_rs::<cubecl::wgpu::WgpuRuntime>(
        image_shape,
        image,
        filter_shape,
        filter,
        os,
        out,
    ));
}
