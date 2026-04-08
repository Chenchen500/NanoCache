/**
 * NanoCache llama.cpp 集成适配器
 *
 * 将 NanoCache Offload Manager 接入 llama.cpp 的 ggml_backend_sched_eval_callback
 * 调度层。每次 tensor 计算前后触发回调，实现 offload/recall 逻辑。
 *
 * 使用方式:
 *   params.cb_eval        = nanocache_cb_eval;       // 核心回调
 *   params.cb_eval_user_data = &g_nc;                // 传入 manager
 *   params.warmup          = false;                   // 跳过 warmup（NanoCache 自己管理）
 *
 * 输出说明:
 *   NanoCache 日志 → stderr
 *   生成文本       → stdout
 */

#include "nanocache_offload.h"
#include "llama.h"
#include "common.h"

#include <locale>
#include <vector>
#include <cstdio>

// ============================================================
// 主循环
// ============================================================

static bool run(llama_context* ctx, const common_params& params) {
    const llama_model* model   = llama_get_model(ctx);
    const llama_vocab* vocab   = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        fprintf(stderr, "%s: no input tokens\n", __func__);
        return false;
    }

    const int n_tokens = (int)tokens.size();
    fprintf(stderr, "[NanoCache] Processing %d tokens...\n", n_tokens);

    // 预热: 注册所有 KV tensor（一次性）
    g_nc.on_decode_start();

    // 分段 decode（长 prompt 场景）
    const int chunk_size = 256;
    int pos = 0;
    int generated = 0;

    while (pos < n_tokens && generated < params.n_predict) {
        int end = std::min(pos + chunk_size, n_tokens);
        int batch_count = end - pos;

        // Decode
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, batch_count))) {
            fprintf(stderr, "%s: llama_decode failed at pos=%d\n", __func__, pos);
            return false;
        }

        pos = end;
        g_nc.on_decode_done();  // 每段结束后统一 offload

        // 采样一个 token
        llama_token new_token = llama_sampling_sample(
            ctx, nullptr, nullptr, common_sampling_params());
        llama_sampling_accept(ctx, new_token, true);

        char buf[128];
        int n = llama_token_to_piece(ctx, new_token, buf, sizeof(buf));
        if (n > 0) {
            printf("%.*s", n, buf);
            fflush(stdout);
            generated++;
        }

        // Decode 开始前 recall
        g_nc.on_decode_start();
    }

    printf("\n");
    return true;
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // 初始化 NanoCache CUDA
    if (!g_nc.init()) {
        fprintf(stderr, "[NanoCache] Warning: CUDA init failed, running without offload\n");
    }

    // 解析 llama.cpp 参数
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // 注入 NanoCache callback
    params.cb_eval        = nanocache_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup          = false;  // NanoCache 自己管理预热

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    auto* model = init->model();
    auto* ctx   = init->context();

    if (!model || !ctx) {
        fprintf(stderr, "failed to initialize model\n");
        return 1;
    }

    // 打印系统信息
    {
        auto info = common_params_get_system_info(params);
        fprintf(stderr, "\n%s\n", info.c_str());
    }

    bool ok = run(ctx, params);
    g_nc.print_stats();

    llama_perf_context_print(ctx);
    llama_backend_free();

    return ok ? 0 : 1;
}
