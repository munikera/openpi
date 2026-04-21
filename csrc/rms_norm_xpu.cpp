/*
 * rms_norm_xpu.cpp — Fused bf16 RMSNorm SYCL kernel for Intel XPU
 * ================================================================
 *
 * Replaces the 6-copy_ Python sequence:
 *   x.float() → variance → rsqrt → scale → weight.float() → .to(bf16)
 * with a single SYCL dispatch that does everything inside one kernel
 * using fp32 registers for accumulation, reading/writing bf16.
 *
 * Also provides fused_ada_rms_norm_xpu which handles the AdaRMSNorm
 * (normalize + linear(cond) → scale/shift/gate) in one dispatch.
 *
 * Build with build_rms_norm.py, then import as:
 *   from scripts.build_rms_norm import ext
 *   out = ext.rms_norm_xpu(x, weight, eps)
 *
 * Expected tensor layouts:
 *   rms_norm_xpu:
 *     x       : [*, H]  bf16, contiguous  (flattened to [N, H])
 *     weight  : [H]     fp32, contiguous
 *     out     : [*, H]  bf16
 *
 *   ada_rms_norm_xpu:
 *     x       : [B, S, H]  bf16, contiguous
 *     weight  : [H]        fp32, contiguous  (RMSNorm scale, 1+weight applied)
 *     mod     : [B, 1, H*3] bf16             (pre-computed linear(cond) output)
 *     out     : [B, S, H]  bf16
 *     gate    : [B, 1, H]  bf16
 *     shift   : [B, 1, H]  bf16
 */

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

// at::BFloat16 is a uint16_t wrapper with float conversion operators.
// It works in SYCL device code — no sycl::bfloat16 needed.
using bf16 = at::BFloat16;

// ── Kernel 1: plain RMSNorm ───────────────────────────────────────────────────

template<int BLOCK>
class RMSNormKernel {
    const bf16*                    x_;
    const float*                   w_;
    bf16*                          out_;
    float                          eps_;
    int                            H_;
    sycl::local_accessor<float, 1> local_;

public:
    RMSNormKernel(const bf16* x, const float* w, bf16* out,
                  float eps, int H, sycl::local_accessor<float,1> local)
        : x_(x), w_(w), out_(out), eps_(eps), H_(H), local_(local) {}

    void operator()(sycl::nd_item<1> item) const {
        const int row = item.get_group(0);
        const int tid = item.get_local_id(0);

        const bf16* xr   = x_   + (size_t)row * H_;
              bf16* outr = out_ + (size_t)row * H_;

        // accumulate sum-of-squares in fp32
        float sum = 0.f;
        for (int i = tid; i < H_; i += BLOCK) {
            float v = float(xr[i]);
            sum += v * v;
        }

        // tree reduction in local memory
        local_[tid] = sum;
        sycl::group_barrier(item.get_group());
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) local_[tid] += local_[tid + s];
            sycl::group_barrier(item.get_group());
        }

        const float rms = sycl::rsqrt(local_[0] / float(H_) + eps_);

        // scale and write
        for (int i = tid; i < H_; i += BLOCK) {
            float v = float(xr[i]);
            outr[i] = bf16(v * rms * (1.0f + w_[i]));
        }
    }
};

// ── Kernel 2: AdaRMSNorm ─────────────────────────────────────────────────────

template<int BLOCK>
class AdaRMSNormKernel {
    const bf16*                    x_;
    const bf16*                    mod_;   // [B, H*3]
    bf16*                          out_;
    bf16*                          gate_;
    float                          eps_;
    int                            S_;
    int                            H_;
    sycl::local_accessor<float, 1> local_;

public:
    AdaRMSNormKernel(const bf16* x, const bf16* mod,
                     bf16* out, bf16* gate,
                     float eps, int S, int H,
                     sycl::local_accessor<float,1> local)
        : x_(x), mod_(mod), out_(out), gate_(gate),
          eps_(eps), S_(S), H_(H), local_(local) {}

    void operator()(sycl::nd_item<1> item) const {
        const int row = item.get_group(0);
        const int tid = item.get_local_id(0);
        const int b   = row / S_;

        const bf16* xr    = x_    + (size_t)row * H_;
        const bf16* modr  = mod_  + (size_t)b   * H_ * 3;
        const bf16* scale = modr;
        const bf16* shift = modr + H_;
        const bf16* gmod  = modr + (size_t)H_ * 2;
              bf16* outr  = out_  + (size_t)row * H_;
              bf16* gater = gate_ + (size_t)row * H_;

        // sum of squares
        float sum = 0.f;
        for (int i = tid; i < H_; i += BLOCK) {
            float v = float(xr[i]);
            sum += v * v;
        }
        local_[tid] = sum;
        sycl::group_barrier(item.get_group());
        for (int s = BLOCK/2; s > 0; s >>= 1) {
            if (tid < s) local_[tid] += local_[tid + s];
            sycl::group_barrier(item.get_group());
        }
        const float rms = sycl::rsqrt(local_[0] / float(H_) + eps_);

        // norm + adaptive scale/shift  (no learned weight — scale comes from mod)
        for (int i = tid; i < H_; i += BLOCK) {
            float v  = float(xr[i]) * rms;
            float sc = float(scale[i]);
            float sh = float(shift[i]);
            outr[i]  = bf16(v * (1.f + sc) + sh);
            gater[i] = gmod[i];
        }
    }
};

// ── Python-facing functions ───────────────────────────────────────────────────

torch::Tensor rms_norm_xpu(
    torch::Tensor x,        // [*, H] bf16
    torch::Tensor weight,   // [H]    fp32
    double        eps_d
) {
    float eps = static_cast<float>(eps_d);

    TORCH_CHECK(x.device().type() == c10::DeviceType::XPU,
                "rms_norm_xpu: x must be on XPU");
    TORCH_CHECK(x.scalar_type()      == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat,    "weight must be fp32");
    TORCH_CHECK(x.is_contiguous(),      "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int64_t N = x.numel() / x.size(-1);
    const int     H = static_cast<int>(x.size(-1));
    TORCH_CHECK(H == weight.numel(), "weight size mismatch");

    auto out   = torch::empty_like(x);
    auto queue = c10::xpu::getCurrentXPUStream().queue();

    auto xp  = x.data_ptr<at::BFloat16>();
    auto wp  = weight.data_ptr<float>();
    auto op  = out.data_ptr<at::BFloat16>();

    constexpr int BLOCK = 256;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local(BLOCK, h);
        h.parallel_for(
            sycl::nd_range<1>(N * BLOCK, BLOCK),
            RMSNormKernel<BLOCK>(xp, wp, op, eps, H, local)
        );
    });

    return out;
}

// ada_rms_norm_xpu:
//   x   : [B, S, H]  bf16
//   mod : [B, H*3]   bf16  (pre-computed dense(cond): scale|shift|gate)
//   returns: (out [B,S,H], gate [B,S,H]) both bf16
//   NOTE: no learned weight — AdaRMS uses dense(cond) output only.
std::tuple<torch::Tensor, torch::Tensor> ada_rms_norm_xpu(
    torch::Tensor x,        // [B, S, H] bf16
    torch::Tensor mod,      // [B, H*3]  bf16
    double        eps_d
) {
    float eps = static_cast<float>(eps_d);

    TORCH_CHECK(x.device().type() == c10::DeviceType::XPU);
    TORCH_CHECK(x.scalar_type()   == torch::kBFloat16);
    TORCH_CHECK(mod.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(mod.is_contiguous());
    TORCH_CHECK(x.dim() == 3, "x must be [B, S, H]");

    const int B = x.size(0);
    const int S = x.size(1);
    const int H = x.size(2);
    TORCH_CHECK(mod.size(0) == B && mod.size(1) == H * 3,
                "mod must be [B, H*3]");

    auto out  = torch::empty_like(x);
    auto gate = torch::empty({B, S, H}, x.options());

    auto queue = c10::xpu::getCurrentXPUStream().queue();
    auto xp = x.data_ptr<at::BFloat16>();
    auto mp = mod.data_ptr<at::BFloat16>();
    auto op = out.data_ptr<at::BFloat16>();
    auto gp = gate.data_ptr<at::BFloat16>();

    constexpr int BLOCK = 256;
    const int N = B * S;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local(BLOCK, h);
        h.parallel_for(
            sycl::nd_range<1>(N * BLOCK, BLOCK),
            AdaRMSNormKernel<BLOCK>(xp, mp, op, gp, eps, S, H, local)
        );
    });

    return {out, gate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_xpu",     &rms_norm_xpu,
          "Fused bf16 RMSNorm — zero copy_ (weight fp32, applied as 1+w)");
    m.def("ada_rms_norm_xpu", &ada_rms_norm_xpu,
          "Fused bf16 AdaRMSNorm — zero copy_ (no learned weight, scale from mod)");
}
