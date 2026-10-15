# Native normalization ordering evidence

The initial device screen (`norm-quant-recovery.json`) failed the unchanged exact
gate. Using a BF16-rounded residual before normalization gave exact residual and
scale, but rare normalized values and INT8 outputs still differed. Those outputs
are not accepted as correct or installed into serving.

Read-only inspection of the deployed `lightop/op.cpython-310-x86_64-linux-gnu.so`
gave these concrete findings:

* `nm -D -C` exposes `at::native::gemma_fused_add_rmsnorm` at address `0x16cdda0`.
* Host disassembly at that address loads the weight-bias argument and jumps to
  `at::native::fused_add_rms_norm_opt_impl<true,false>` through its PLT entry.
* Exported device-kernel symbols include
  `vllm::fused_add_rms_norm_kernel<c10::BFloat16,8,true,false>` and the width-zero
  specialization. This gives evidence for a vector-width-8/block-reduction path;
  it does not prove which specialization or reduction tree a given launch uses.
* Host dispatch contains size/alignment branches and 256/1024-thread constants.
  We have not disassembled the device binary to establish every FP32 operation,
  and no fresh GPU profiler was run while the model pipeline test was active.

The related local vLLM `csrc/libtorch_stable/layernorm_kernels.cu` uses per-thread
eight-value accumulation followed by a CUB block reduction, and an ordinary
left-associated output multiplication. That source is a useful structural clue,
not proof that the vendor binary is the same revision or arithmetic sequence.

`probe_norm_order.py` tests exactly eight named hypotheses: flat reduction with
two final multiplication associations and fused/separate epsilon arithmetic,
plus vector-width-8 per-thread accumulation with/without explicit FMA and the
two final associations. It holds the observed BF16 residual rounding, FP32
gamma addition, BF16 norm output, and deployed quantizer constant. It compares
all norm values, INT8 outputs, and scales to the native operation and records
specific differing inputs. It does not modify the existing fusion helper, run
performance measurements, or relax its exact gate.

Root may run the bounded probe after reserving GPU4:

```sh
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 timeout 180s \
python3 /bench/patch/ttft_norm_quant/probe_norm_order.py --tokens 4096 \
  --output /bench/result/ttft/issue3-attention/norm-order-probe.json
```

An exact result at 4096 tokens only identifies a hypothesis for further exact
checks; it does not replace checks at all shapes and seeds. If none match, stop
this bounded route until native device arithmetic or source is available.
Blindly extending a parameter sweep or accepting a small nonzero mismatch would
not establish equivalence.
