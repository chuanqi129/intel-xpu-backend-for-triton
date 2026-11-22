// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=tt.descriptor_load

// COM: The width keeps its whole granules; the dropped column is loaded where
// COM: the tile reaches it and selected into the tile (#8814).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK: #[[$LANES:.*]] = #ttg.linear<{register = [], lane = {{\[\[1\], \[2\], \[4\], \[32\]\]}}, warp = {{\[\[0\], \[8\], \[16\]\]}}, block = []}>
  // CHECK-LABEL: tt.func @unaligned_width
  tt.func @unaligned_width(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %x: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: %[[WIDTH:.*]] = arith.trunci
    // CHECK: %[[GRANULES:.*]] = arith.andi %[[WIDTH]], %c-2_i32
    // CHECK: %[[BW:.*]] = arith.muli %[[GRANULES]], %c2_i32
    // CHECK: %[[TILE:.*]] = ttig.2d_block_load %{{.*}}, %[[BW]], {{.*}} {row_major}
    // CHECK: %[[COLUMN:.*]] = scf.if %{{.*}} -> (tensor<64xf16, #ttg.slice<{dim = 1, parent = #{{.*}}}>>) {
    // CHECK:   tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f16>, #[[$LANES]]>
    // CHECK:   ttg.convert_layout %{{.*}} : tensor<64xf16, #[[$LANES]]> -> tensor<64xf16, #ttg.slice
    // CHECK: } else {
    // CHECK: %[[IS_COL:.*]] = arith.cmpi eq
    // CHECK: %[[COL:.*]] = tt.broadcast %{{.*}} : tensor<64x1xf16, #{{.*}}> -> tensor<64x32xf16, #{{.*}}>
    // CHECK: arith.select %[[IS_COL]], %[[COL]], %[[TILE]]
    %0 = tt.descriptor_load %desc[%c0_i32, %x] {ttig.block_io = "row_major", ttig.block_io_unaligned_width} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: i8 drops up to three columns; the signed block type loads them through
// COM: a signless pointer.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @unaligned_width_i8
  tt.func @unaligned_width_i8(%arg0: !tt.ptr<i8>, %arg1: i32, %arg2: i32, %arg3: i64, %x: i32) -> tensor<64x64xi8, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <i8>, <64x64xsi8>
    // CHECK: arith.andi %{{.*}}, %c-4_i32
    // CHECK: %[[TILE:.*]] = ttig.2d_block_load
    // CHECK: scf.if
    // CHECK: tt.bitcast %{{.*}} : !tt.ptr<si8> -> !tt.ptr<i8>
    // CHECK-COUNT-3: ttg.convert_layout
    // CHECK: %[[S0:.*]] = arith.select %{{.*}}, %{{.*}}, %[[TILE]]
    // CHECK: %[[S1:.*]] = arith.select %{{.*}}, %{{.*}}, %[[S0]]
    // CHECK: arith.select %{{.*}}, %{{.*}}, %[[S1]]
    %0 = tt.descriptor_load %desc[%c0_i32, %x] {ttig.block_io = "row_major", ttig.block_io_unaligned_width} : !tt.tensordesc<64x64xsi8> -> tensor<64x64xi8, #dot0>
    tt.return %0 : tensor<64x64xi8, #dot0>
  }
}
