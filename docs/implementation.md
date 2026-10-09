# Kernel implementation notes

## Tensor contract

Input tensors use contiguous B/H/S/D layout. The extension requires Q, K, and V to have identical shape, dtype, and CUDA device. Supported dtypes are FP32 and FP16; head dimension must be in [1, 128]. The output has the same shape and dtype as Q.

The QK scale is 1/sqrt(D). Causal masking uses the inclusive lower triangle: query position i can attend to key positions j <= i.

## Tiling and shared memory

Each CUDA block owns one query row. The block stages at most 32 K and V rows in shared memory, with a fixed shared-memory stride of 128 elements per row. Threads cooperatively load the K/V tile. The first 32 threads calculate scores, one score per key, using FP32 accumulation.

This design makes the data movement visible and keeps the control flow approachable. It is intentionally not an optimized implementation: score dot products and the online output update contain serial loops, and only thread 0 updates the row output accumulator. Production kernels parallelize work across query rows/features, optimize memory transactions, and may use architecture-specific matrix instructions.

## Online softmax

For one query row, let the state after previously processed tiles be maximum m, denominator l, and unnormalized output accumulator o. For scores s from the current tile:

- m_new = max(m, max(s))
- alpha = exp(m - m_new)
- p = exp(s - m_new)
- l_new = alpha*l + sum(p)
- o_new = alpha*o + sum(p*V)

The final output is o/l. Initially m=-infinity, l=0, and o=0. Masked and out-of-range keys receive negative-infinity scores and zero probability.

## Numerical behavior

Input values are cast to FP32 for dot products and updates; the result is cast back to the input dtype. Results need not be bitwise identical to SDPA because floating-point reductions and fused kernels may use different operation orders. Test tolerances are starting points and must be validated on the target architecture and PyTorch version.

## Further optimization ideas

1. Let warps cooperate on QK dot products and parallelize the output-feature update.
2. Process multiple query rows per CTA to amortize K/V loads.
3. Use vectorized global memory access and tune tile sizes.
4. Tune launch geometry and inspect registers, occupancy, and shared-memory pressure with Nsight Compute.
5. Add backward support and broader shape semantics only after forward correctness is stable.
