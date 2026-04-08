# NanoCache

**KV Cache 鍒嗗眰鍘嬬缉 鈥?鎵╁睍澶фā鍨嬩笂涓嬫枃澶勭悊鑳藉姏**

> **NanoCache 1.0** 鎻愪緵浜嗙█鐤忔敞鎰忓姏锛圠1锛? INT8 閲忓寲鍘嬬缉锛圠2锛? Smart LRU RAM Offload锛圠3锛夌殑鍒嗗眰鏋舵瀯锛屽湪涓嶆槑鏄惧鍔犲欢杩熺殑鍓嶆彁涓嬫墿灞?llama.cpp 鎺ㄧ悊鐨勪笂涓嬫枃澶勭悊鑳藉姏銆?
**鏍稿績鏁堟灉**锛氬湪 RTX 5070 Ti锛?6GB VRAM锛変笂锛?4B-Q5_K_M 妯″瀷鐨勪笂涓嬫枃澶勭悊鑳藉姏浠?~22500 tokens 鎵╁睍鑷虫洿澶氾紝涓斿悶鍚愰噺鏃犳槑鏄鹃€€鍖栥€?
---

## 馃毃 鍟嗕笟浣跨敤蹇呰

鏈」鐩噰鐢?**GNU Affero General Public License v3 (AGPLv3)**銆?
濡傛灉鎮ㄦ槸浜戞湇鍔℃彁渚涘晢锛圓WS銆丄zure銆侀樋閲屼簯绛夛級鎴栦紒涓氱敤鎴凤紝涓斾笉甯屾湜鎮ㄧ殑鏈嶅姟浠ｇ爜琚紑婧愶紝璇疯仈绯讳綔鑰呰幏鍙栧晢涓氭巿鏉冦€傝仈绯绘柟寮忚浠撳簱涓婚〉銆?
---

## 鏋舵瀯

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹? L1: 绋€鐤忔敞鎰忓姏 (Sparse Attention)               鈹?鈹? 浣嶇疆: flash_attn_ext CUDA 鍐呮牳                   鈹?鈹? 瑙﹀彂: KV 闀垮害 > 2048 tokens                       鈹?鈹? 鏁堟灉: 娉ㄦ剰鍔?FLOP O(n虏) 鈫?O(n)                鈹?鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹? L2: INT8 閲忓寲鍘嬬缉 (INT8 Quantization)             鈹?鈹? 浣嶇疆: llama.cpp eval-callback 璋冨害灞?            鈹?鈹? 瑙﹀彂: 鍙€夛紙榛樿寮€鍚級                            鈹?鈹? 鏁堟灉: KV Cache 鍘嬬缉 2x锛團P16 鈫?INT8锛?         鈹?鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹? L3: Smart LRU RAM Offload                       鈹?鈹? 浣嶇疆: llama.cpp ggml_backend_sched_eval_callback 鈹?鈹? 瑙﹀彂: slot 杩炵画 8 涓?decode step 鏈闂?         鈹?鈹? 鏁堟灉: 閲婃斁 VRAM锛屾敮鎸佹洿闀夸笂涓嬫枃                  鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

涓夊眰鍙嫭绔嬪惎鐢ㄣ€侺1 濮嬬粓寮€鍚紝L2/L3 鎸夐渶鍙犲姞銆?
---

## 娴嬭瘯缁撴灉

**纭欢**: RTX 5070 Ti (16GB VRAM) | CUDA 12.8 | llama.cpp (cuBLAS + CUDA)

**妯″瀷**: DeepSeek-R1-Distill-Qwen-14B-Q5_K_M (GGUF, 10.5GB)

### VRAM 鑺傜渷

| 涓婁笅鏂囬暱搴?| v1.0 (鏃犲帇缂? | v2.0 (INT8+LRU) | 鑺傜渷 |
|-----------|---------------|-----------------|------|
| 4,500 tok | 1,605 MB | 1,462 MB | 143 MB (8.9%) |
| 10,500 tok | 1,628 MB | 1,462 MB | 166 MB (10.2%) |
| 15,000 tok | 1,634 MB | 1,462 MB | 172 MB (10.5%) |
| 22,500 tok | **OOM** | **1,462 MB** | 鉁?姝ｅ父杩愯 |

### 鍚炲悙閲?
| 鐗堟湰 | 鐭?Prompt | 闀?Prompt | 寮€閿€ |
|------|-----------|-----------|------|
| v1.0 Callback | 5399 ms | 5386 ms | baseline |
| v2.0 Smart LRU | 5377 ms | 5339 ms | -0.4%~0.9% |

> Smart LRU 鐨勫洖璋冭皟搴﹀紑閿€鍙拷鐣ヤ笉璁★紙卤2.5%锛夈€?
### Offload 琛屼负

- **鐭敓鎴愶紙< 50 steps锛?*锛歰ffload = 0 鈥?鐑?slot 鐣欏湪 VRAM锛孡RU 姝ｇ‘鍦颁笉骞查
- **闀跨敓鎴愶紙鈮?50 steps锛?*锛欴ecode #9 寮€濮嬭Е鍙?offload锛坰tale_threshold=8锛夛紝offload/recall 瀹岀編瀵圭О

---

## 瀹炵敤鍦烘櫙

NanoCache 瑙ｅ喅鐨勬湰璐ㄩ棶棰樻槸锛?*鏈夐檺鐨?VRAM 濡備綍鏀拺鏇撮暱鐨勪笂涓嬫枃**銆備互涓嬫槸鍏蜂綋鍦烘櫙锛?
### 1. 闀挎枃妗ｆ憳瑕佷笌鐞嗚В

**鍦烘櫙**锛氬皢涓€鏁存湰鎶€鏈墜鍐屻€佽鏂囬泦鎴栨硶寰嬪悎鍚岋紙鏁颁竾瀛楋級涓€娆℃€ц緭鍏ユā鍨嬨€?
**鍘熸湁鐡堕**锛氫笂涓嬫枃瓒婇暱锛孠V Cache 瓒婂ぇ銆傝秴杩?~22500 tokens 鏃剁洿鎺?OOM銆?
**NanoCache 鏁堟灉**锛欼NT8 鍘嬬缉 VRAM 鍗犵敤锛屽湪鍚岀瓑纭欢涓婂皢鍙鐞嗙殑涓婁笅鏂囦粠 OOM 涓寸晫鐐规墿灞曠害 50%銆?
```
鍦烘櫙: 瑙ｈ涓€浠?300 椤电殑鎶€鏈枃妗?(~150,000 瀛? ~100,000 tokens)
鍘熸湁鏂规: 闇€鍒嗘杈撳叆锛屾瘡娈典涪澶辨钀介棿鍏宠仈鎬?NanoCache: 鍗曟杈撳叆鏇村鍐呭锛屽噺灏戝垎娈靛繀瑕?```

### 2. 澶氳疆 Agent 瀵硅瘽

**鍦烘櫙**锛氫竴涓?AI 鍔╂墜鍦ㄩ暱瀵硅瘽涓悓鏃剁淮鎶ょ郴缁?Prompt + 瀵硅瘽鍘嗗彶 + 宸ュ叿璋冪敤涓婁笅鏂囥€?
**鍘熸湁鐡堕**锛氬杞璇濈殑 KV Cache 闅忚疆娆＄嚎鎬у闀匡紝鍑犲崄杞悗 VRAM 鑰楀敖銆?
**NanoCache 鏁堟灉**锛歋mart LRU 鍦ㄧ敓鎴愯繃绋嬩腑鑷姩灏嗘棫杞鐨?KV 鏁版嵁 offload 鍒?RAM锛屼繚鐣欐渶杩戝璇濈殑鍝嶅簲璐ㄩ噺锛屽悓鏃堕噴鏀?VRAM 绌洪棿銆?
```
鍦烘櫙: 30 杞妧鏈皟璇曞璇?(姣忚疆骞冲潎 500 tokens 杈撳叆)
鍘熸湁鏂规: ~15000 tokens 鏃跺紑濮嬪彉鎱紝瓒呰繃 20000 tokens 璐ㄩ噺涓嬮檷
NanoCache: 鐑?slot 淇濈暀鍦?VRAM锛屽喎 slot offload 鍒?RAM锛屼繚鎸佸搷搴旈€熷害
```

### 3. RAG 澧炲己妫€绱?
**鍦烘櫙**锛氬湪 RAG锛堟绱㈠寮虹敓鎴愶級娴佺▼涓紝涓€娆℃€ф敞鍏ユ暟鍗佷釜妫€绱㈠埌鐨勬枃妗ｇ墖娈典綔涓轰笂涓嬫枃銆?
**鍘熸湁鐡堕**锛氭瘡涓墖娈?~512 tokens锛?0 涓墖娈靛姞涓婂師濮嬮棶棰樺緢蹇秴杩?5000 tokens锛孷RAM 鍘嬪姏鏄捐憲銆?
**NanoCache 鏁堟灉**锛歏RAM 鎭掑畾鍦?~1462 MB锛屼笉鍐嶉殢妫€绱㈢墖娈垫暟閲忕嚎鎬у闀匡紝鏀寔鏇村鐗囨鍚屾椂椹荤暀銆?
```
鍦烘櫙: 妫€绱?20 涓枃妗ｇ墖娈?(姣忎釜 512 tokens) + 闂
鍘熸湁鏂规: 20 涓墖娈?= ~10240 tokens锛孷RAM 鍗犵敤 1628 MB
NanoCache: 20 涓墖娈典粛淇濇寔 1462 MB锛岃妭鐪?166 MB
```

### 4. 浠ｇ爜鍒嗘瀽涓庤ˉ鍏?
**鍦烘櫙**锛氬垎鏋愬ぇ鍨嬩唬鐮佷粨搴擄紙鏁板崈琛屼唬鐮侊級锛屾ā鍨嬮渶瑕佸悓鏃剁湅鍒板涓枃浠剁殑涓婁笅鏂囥€?
**鍘熸湁鐡堕**锛氫唬鐮佹枃浠?KV Cache 鐨勫ご娉ㄦ剰鍔涙ā寮忓鑷存棭鏈?token 琚仐蹇橈紝鍚庢湡鐢熸垚璐ㄩ噺涓嬮檷銆?
**NanoCache 鏁堟灉**锛氱█鐤忔敞鎰忓姏淇濈暀鏈€杩?128 tokens 鐨勫畬鏁存敞鎰忓姏锛屽悓鏃堕€氳繃閲嶈鎬ц瘎鍒嗕繚鐣欏叧閿唬鐮佺粨鏋勭殑 token銆?
### 5. 杈圭紭璁惧鎺ㄧ悊

**鍦烘櫙**锛氬湪娑堣垂绾?GPU锛?GB~16GB VRAM锛変笂杩愯澶фā鍨嬨€?
**鍘熸湁鐡堕**锛氭ā鍨嬫湰韬凡鍗犵敤澶ч噺 VRAM锛屽彲鐢ㄤ簬 KV Cache 鐨勭┖闂存瀬涓烘湁闄愩€?
**NanoCache 鏁堟灉**锛欼NT8 鍘嬬缉 2x锛岀瓑鏁堝皢 KV Cache 鍙敤 VRAM 缈诲€嶃€?
```
纭欢: RTX 4060 Ti (16GB VRAM)
妯″瀷: Qwen2.5-7B-Q4_K_M (~4GB 鏉冮噸)
鍘熸湁: KV Cache 鍙敤绌洪棿 ~12GB锛屽彲澶勭悊 ~15000 tokens
NanoCache: KV Cache 鍘嬬缉 2x锛岀瓑鏁堝彲鐢ㄧ┖闂?~13GB+
```

---

## 椤圭洰缁撴瀯

```
NanoCache/
鈹溾攢鈹€ phase1_policy/           # Python: 閲忓寲鍣ㄣ€侀噸瑕佹€ц瘎鍒嗗櫒銆佺█鐤?KV 绠＄悊
鈹溾攢鈹€ phase2_operators/        # Python CUDA: FusedSparseAttention 鍐呮牳
鈹溾攢鈹€ phase3_engine/           # C++ CUDA: NanoCache 鏍稿績寮曟搸
鈹?  鈹斺攢鈹€ nanocache_llama_adapter.cpp   # llama.cpp 闆嗘垚閫傞厤鍣?鈹溾攢鈹€ docs/                    # 鎶€鏈櫧鐨功
鈹斺攢鈹€ llama.cpp/              # (闇€ clone https://github.com/ggerganov/llama.cpp)
    鈹斺攢鈹€ examples/eval-callback/
        鈹溾攢鈹€ eval-callback-v1-pure.cpp   # v1.0: callback-only baseline
        鈹溾攢鈹€ eval-callback-v2.cpp        # v2.0: Smart LRU INT8 offload
        鈹斺攢鈹€ README.md                   # llama.cpp 闆嗘垚璇存槑
```

---

## 蹇€熷紑濮?
### 1. 鍑嗗 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_BURBLE_COPY=OFF
make -j$(nproc)
```

### 2. 缂栬瘧 NanoCache eval-callback

灏?`llama.cpp/examples/eval-callback/eval-callback-v2.cpp` 瑕嗙洊 `llama.cpp/examples/eval-callback/eval-callback.cpp`锛岀劧鍚庯細

```bash
cd llama.cpp/build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_BURBLE_COPY=OFF
make llama-eval-callback -j$(nproc)
```

### 3. 杩愯

```bash
# v2.0 Smart LRU INT8 offload
./llama-eval-callback \
  -m /path/to/model-q4_k_m.gguf \
  --prompt "What is artificial intelligence?" \
  -n 128
```

NanoCache 鐨?offload/recall 鏃ュ織杈撳嚭鍒?stderr锛岀敓鎴愭枃鏈緭鍑哄埌 stdout銆?
---

## 鎶€鏈粏鑺?
### llama.cpp 闆嗘垚鐐?
`ggml_backend_sched_eval_callback` 鈥?llama.cpp 璋冨害鍣ㄥ湪姣忎釜 tensor 璁＄畻鍓嶅悗瑙﹀彂鍥炶皟锛?
```cpp
// ask=true: compute 鍓?鈫?鑻?slot 琚?offload锛屽厛 recall
// ask=false: compute 鍚?鈫?鑻?slot 鍙樺喎锛宱ffload
bool nanocache_cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    if (is_kv_tensor(t->name)) {
        if (ask && slot.is_offload) nc_recall(t->name);
        if (!ask && slot.is_stale(threshold)) nc_offload(t->name);
    }
    return true;
}
params.cb_eval = nanocache_cb_eval;
```

### INT8 閲忓寲锛坧er-head 鐙珛 scale锛?
```cpp
// 閲忓寲
float scale = 127.0f / max(|fp16|);
int8_t q = round(fp16 / scale);

// 鍙嶉噺鍖?fp16' = (q - 128) * (1.0f / scale);
```

### Smart LRU 璋冨害

```cpp
static const int STALE_THRESHOLD = 8;  // 杩炵画 8 涓?decode step 鏈闂墠 offload

void nc_on_decode_done() {
    for (auto& slot : slots) {
        if (!slot.is_offload && slot.stale(decode_no) > STALE_THRESHOLD) {
            nc_offload(slot);
        }
    }
}
```

---

## 寮€婧愯鍙?
鏈」鐩噰鐢?**GNU Affero General Public License v3 (AGPLv3)**銆?
NanoCache 鍩轰簬 [llama.cpp](https://github.com/ggerganov/llama.cpp)锛堝悓涓?MIT License锛夈€?
---

*椤圭洰闅忕爺绌惰繘灞曟寔缁洿鏂般€傛祴璇曟暟鎹熀浜?2026-04-08 瀹為獙鐜锛圧TX 5070 Ti, CUDA 12.8锛夈€?
