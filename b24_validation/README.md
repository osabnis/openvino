# B24: OpenVINO used for the llama.cpp B70 validation runs

This branch is the OpenVINO source of build B24, used for the 52-model llama.cpp OpenVINO-backend validation runs on
Arc Pro B70 (run 24, 2026-11-24, and run 25, 2026-11-26).

## Contents
Base: openvinotoolkit/openvino master `05751a037d`, with these PR heads merged (one merge commit each):

| PR | Head commit used |
|---|---|
| #38696 | `c38e757f1d` |
| #39930 (includes #39710) | `de4351fc5c` |
| #39594 | `8761ae0b1e` |
| #40083 | `eae475f273` |
| #39555 | `160754826e` |

Plus two changes that are not in a PR yet (in the last commit of this branch):
- GPU fused MoE: oneDNN grouped matmul static zero point (fused-MoE flag): `moe_3gemm_fused_compressed.hpp`,
  `prepare_quantization.cpp`, `moe_3gemm_swiglu_opt.cpp/.hpp`
- GPU plugin memory fix: release dynamic output memory to the pool, and the output chain stops at runtime-skippable
  nodes: `primitive_inst.cpp`, `network.cpp`

And one oneDNN change, which lives in the `onednn_gpu` submodule and is therefore shipped as a patch:
`b24_validation/onednn_gpu_sbzp.diff` (runtime part of oneDNN #6560, `src/gpu/intel/matmul/grouped_micro_gemm.cpp`).

## Build
```
git submodule update --init --recursive
git -C src/plugins/intel_gpu/thirdparty/onednn_gpu apply "$PWD/b24_validation/onednn_gpu_sbzp.diff"
```
Then build OpenVINO as usual (B24 was a GPU-plugin build).

## Driver note for Arc Pro B70
compute-runtime 26.31.39395.13 and 26.40.40196.8 can map two 2 MiB pages of one USM device allocation to the same
physical memory, which silently corrupts large weights (seen on Muse-Glimmer-30B's LM head). Workaround used in run 25:
`NEOReadDebugKeys=1 EnableDeviceUsmAllocationPool=0` in the environment of every GPU run.
