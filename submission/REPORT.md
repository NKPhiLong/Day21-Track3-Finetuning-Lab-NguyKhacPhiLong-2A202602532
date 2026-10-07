# Lab 21 — Evaluation Report

**Họ tên**: Nguỵ Khắc Phi Long  **MSSV**: 2A202602532  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14,6 GB khả dụng, sm_75 → fp16)

> Mọi con số dưới đây lấy từ `results/` (commit `d27c1c0`, chạy đủ — `eval_limit = null`,
> `smoke_mode = false`). Chạy NB1→NB5 tuần tự trên Colab T4: NB1 21 s · NB2 431 s · NB3 488 s ·
> NB4 1402 s · NB5 718 s ≈ **51 phút**, thấp hơn nhiều so với ước tính 100–130 phút của README.

---

## 1. Setup — lựa chọn và lý do

| | |
|---|---|
| Base model | `unsloth/Qwen3.5-4B` — mặc định của tier T4 |
| Dataset | corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 ticket target + 15 câu regression (checksum không đổi — `make verify` xanh) |
| `max_length` | 1024 (tier T4) — p95 đo được là **98**, max 101, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** optimizer step (batch 1 × grad_accum 16 = batch hiệu dụng 16) |
| LoRA | `text-linear` (12 loại module), r=16, α=32, LR 1e-4 cosine, warmup 3 |

**Vì sao giữ model và dataset mặc định.** Mục tiêu của lab là *đo cho đúng*, không phải
thắng. Corpus mặc định có thang chấm hoàn toàn khách quan (so trường JSON với nhãn), không
cần LLM judge, và đã có mốc đóng băng + checksum để `make verify` kiểm tra — đổi dataset ở
lần chạy đầu sẽ làm mất các lưới an toàn đó. Qwen3.5-4B là model lớn nhất vừa T4 ở bf16/fp16
LoRA (đỉnh VRAM đo được 8,78 GB / 14,6 GB), nên không phải đánh đổi sang QLoRA.

**Về `max_length` (rubric 1.3).** Tôi giữ 1024 của tier thay vì 256 theo p95, và đã kiểm tra
rằng lệch này không ảnh hưởng kết quả: `max_length` chỉ là trần cắt chuỗi, mẫu dài nhất là
101 token nên không mẫu nào bị cắt; `packing=False` và batch = 1 nên cũng không có chi phí
padding. Nếu chạy lại tôi sẽ đặt 256 cho đúng nguyên tắc "đo rồi mới đặt".

**Template có giữ khối `<think>` không?** **Có** — `results/template_check.json`:
`open_tag_present = true`, `body_present = true`, verdict *"reasoning preserved"*. Với corpus
này mọi câu trả lời là JSON trần, và template tự chèn một khối `<think>\n\n</think>` rỗng
vào cuối generation prompt, nên không có trace nào để bảo vệ hay làm hỏng.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) — xa ngưỡng 0.95 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (giải mã ngược các vị trí `labels != -100`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị che: system prompt, toàn bộ lượt `user` và `<|im_start|>assistant\n<think>\n\n`.
Đối chứng `MASK_MODE=everything` cho 94/94 token (100 %) — đúng lỗi kinh điển mà lab cảnh
báo. Trên toàn tập train, NB3 đo lại được 9014 / 20951 token được giám sát (43,0 %), khớp
với mẫu đơn lẻ. Lưu ý: `</think>\n\n` nằm *trong* loss — model học cách tự đóng khối think
rỗng, chấp nhận được vì đó là điều template sẽ làm lúc suy luận.

---

## 3. Ba baseline (NB2 đóng băng TRƯỚC khi train, NB5 chấm (c))

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3431.8 |
| (b) base + optimized prompt | **0.765** | **0.7911** | 1.000 | 1087.5 |
| (c) LoRA fine-tune (`correct`) | **0.970** | 0.6778 | 1.000 | 1465.8 |

*(results/baselines_frozen.json, results/verdict.json)*

**(b) có thật sự mạnh hơn (a) không?** Có, rất rõ: 0.000 → 0.765. Với prompt một dòng
"Phân loại ticket sau.", base model trả lời bằng văn xuôi — `format = 0` nên `target = 0`
theo định nghĩa, và latency cao gấp ~3× vì nó sinh văn bản dài tới trần token. Prompt (b)
(schema + tập giá trị + một ví dụ) đưa `format` lên 1.0 ngay mà không cần train.

**Có sửa `OPTIMIZED_PROMPT` không?** Không. SHA `719e74d3b6232053` khớp với bản gốc
(`make verify`: *baseline (b) prompt unmodified* — ok). (c) được chấm với prompt **naive**
— đúng ý đồ của fine-tune: hành vi chuyển vào trọng số nên prompt được phép ngắn lại.

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

Cả bốn run: cùng 30 step, cùng dữ liệu, cùng seed — mỗi run đổi **đúng một** biến.

| Run | biến bị đổi | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | — | text-linear (12) | 16 | 32 464 896 | 1e-4 | 0.6264 | **0.97** | 425.2 | 8.78 |
| `attn_only` | vị trí | q,v (2) | 283 *(matched)* | 32 456 704 | 1e-4 | **0.5379** | **0.97** | 288.4 | 8.79 |
| `wrong_lr` | learning rate | text-linear (12) | 16 | 32 464 896 | 1e-5 | 1.5702 | 0.00 | 429.0 | 8.78 |
| `qlora` | lượng tử hoá 4-bit | text-linear (12) | 16 | 32 464 896 | 1e-4 | 0.7058 | 0.94 | 508.2 | **3.86** |

*(results/runs.csv, results/autopsy.json)* — `attn_only` lệch ngân sách 0,025 % so với
`correct` (< 5 %, `make verify` xanh).

**Thứ tự theo target:** `correct` = `attn_only` (0.97) > `qlora` (0.94) ≫ `wrong_lr` (0.00).
**Thứ tự theo train loss:** `attn_only` (0.538) < `correct` (0.626) < `qlora` (0.706) < `wrong_lr` (1.570).
Hai thứ tự **khác nhau** ở vị trí đầu: theo loss, `attn_only` "thắng"; theo tác vụ, nó chỉ hoà.

**4.1 — `attn_only` vs `correct`.** Cùng ngân sách ~32,5 M tham số, `attn_only` **hoà**
`correct` trên target (0.97 = 0.97), nhưng có train loss thấp hơn. Nếu xếp hạng bằng cột
loss tôi sẽ kết luận sai rằng "gắn vào attention với rank cao là tốt hơn". Cần đọc đúng cột
`final_loss`: đó là `train_loss` *trung bình cả run* (trainer báo `train_loss` 0.6264 trong
khi loss ở step cuối của `correct` chỉ ~0.03), nên nó phản ánh tốc độ hội tụ ở đầu run
(step 10: `attn_only` 0.827 vs `correct` 1.381), không phải chất lượng điểm cuối. Kết luận về
*vị trí vs rank*: trên tác vụ hẹp này, đổi vị trí khi giữ nguyên ngân sách **không** tạo ra
khác biệt đo được — đáng chú ý là với Qwen3.5 lai, `q_proj`/`v_proj` chỉ tồn tại ở 8 lớp
`full_attention` (24 lớp còn lại là `linear_attention` dùng `in_proj_qkv`…), tức `attn_only`
chỉ chạm 8/32 lớp mà vẫn hoà. Tôi **không** suy ra "vị trí không quan trọng" nói chung:
cả hai run cùng chạm trần 0.97, và cả 6 lỗi còn lại của `correct` là cùng một lỗi (mục 6) —
tập target không đủ khó để phân biệt hai cấu hình. Đòn bẩy thật trong bảng này là **LR**
(0.97 → 0.00), lớn hơn hẳn vị trí (0.00) và lượng tử hoá (−0.03).

**4.2 — `wrong_lr` chỉ khác một con số (1e-5 thay vì 1e-4).** Loss trung bình 1.570 so với
0.626 — tức loss vẫn giảm, chỉ chậm hơn. Nếu chỉ nhìn loss mà không biết LR, tôi có thể kết
luận "model đang học, chỉ cần thêm step" hoặc "dữ liệu khó". Thực tế sau 30 step model **chưa
học được định dạng**: `format = 0.00`, `target = 0.00`, và latency 5622.9 ms vì nó vẫn sinh
văn xuôi dài như baseline (a). Đây là mất mát từ một con số duy nhất — đúng luận điểm §11.3:
LoRA cần LR ~10× full-FT.

**4.3 — `qlora`.** Tiết kiệm **4,92 GB VRAM (8.78 → 3.86, −56 %)**. Trả giá: target −0.03
(0.97 → 0.94, tương đương 6 trường sai thêm trên 200), train chậm hơn 19 % (508 s vs 425 s)
và suy luận chậm hơn 33 % (1951.6 vs 1465.8 ms/mẫu) do dequantize. Số đo **ủng hộ có điều
kiện** khuyến nghị "không dùng QLoRA cho dòng model này": có mất chất lượng thật, nhưng nhỏ
trên tác vụ này. Trên T4, bf16/fp16 LoRA đã vừa (8,78 GB) nên không có lý do để trả giá đó;
QLoRA chỉ đáng khi GPU không chứa nổi bản 16-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.113` (ngưỡng −0.020) · `valid_trace_rate = 0.00`

Bản fine-tune thắng mốc (b) rất rõ trên tác vụ (+0.205, 0.765 → 0.970) với format 1.0 và
prompt ngắn hơn nhiều — nhưng cổng hồi quy đánh trượt vì năng lực phổ thông tụt 0.113, gấp
~5,7 lần ngưỡng cho phép. Để hiểu *vì sao*, tôi chạy thêm `scripts/qual_probe.py` (sinh lại
đúng các prompt của NB2/NB5 với adapter tắt/bật, kết quả ở `results/qualitative_probe.json`;
điểm regression trung bình tái lập chính xác 0.7911 / 0.6778). Nguyên nhân **không phải**
model quên kiến thức. Nó đã **khái quát hoá quá mức định dạng đầu ra**: 14/15 câu hỏi phổ
thông bị trả lời bằng một object JSON (câu còn lại — dịch sang tiếng Anh — ra một chuỗi
JSON có ngoặc kép); (b) thì 0/15. Ở đa số câu, đáp án vẫn nằm trong JSON (`"answer":
"Hà Nội"`) nên keyword recall vẫn đạt; nhưng ở các câu như "1 km bằng bao nhiêu mét?" hay
"Một năm có bao nhiêu tháng?", model áp nguyên schema triage — `{"intent": "hoi_thong_tin",
"urgency": "thap", "product": null, …}` — và không trả lời. 225 mẫu train, mẫu nào cũng là
"user hỏi → JSON triage", nên 2 epoch đủ để dạy model rằng *mọi* lượt assistant đều là JSON.

Hai lưu ý để không đọc quá mức con số: (1) tập regression chỉ có 15 câu, mỗi câu = 0.067
điểm, nên −0.113 tương đương ~1,7 câu; (2) chỉ số là keyword recall với trần 96 token, nên
có nhiễu theo cả hai chiều (bản fine-tune *thắng* câu "2 mũ 10" chỉ vì (b) giải thích dài và
bị cắt trước khi viết ra 1024). Dù vậy, hành vi "JSON hoá mọi thứ" là rõ ràng và là rủi ro
thật khi deploy. Theo thứ tự chẩn đoán của NB5, đây là trường hợp 2 (quên thảm hoạ dạng định
dạng) → cách sửa là trộn 1–5 % dữ liệu hội thoại phổ thông (replay, deck §6.3) vào tập train,
**không** phải nới ngưỡng. `valid_trace_rate = 0.00` là kỳ vọng: suy luận chạy với
`enable_thinking=False` và dữ liệu train không có trace, nên đây không phải bằng chứng về
reasoning-trace collapse (không đo B3).

---

## 6. Định tính — có cả ca THUA

Dự đoán (b) và (c) lấy từ `results/qualitative_probe.json`; điểm target là tỷ lệ trường đúng.

| # | Đầu vào (rút gọn) | Nhãn / kỳ vọng | (b) prompt tối ưu | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| T47 | "…ốp lưng… Shipper không gọi. Hỏi cho biết thôi." | `van_chuyen`, `thap` | `hoi_thong_tin` — 0.75 | `van_chuyen` — **1.00** | ✅ FT thắng: hiểu "shipper không gọi" là vận chuyển |
| T48 | "…ốp lưng… Giá bao nhiêu. Mong shop phản hồi." | `hoi_thong_tin` | `doi_tra` — 0.75 | `hoi_thong_tin` — **1.00** | ✅ FT thắng |
| T0 | "…chuột không dây… Cho tôi trả lại. Gấp." | `doi_tra`, `cao` | `hoan_tien` — 0.75 | `doi_tra` — **1.00** | ✅ FT thắng: tách được đổi trả / hoàn tiền |
| R2 | "1 km bằng bao nhiêu mét?" | có "1000" | "…**1 km = 1000 m**." — 1.0 | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, …}` — **0.0** | ❌ **FT thua**: áp schema triage lên câu hỏi kiến thức |
| R9 | "Một năm có bao nhiêu tháng?" | có "12" | "…có **12 tháng**." — 1.0 | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` — **0.0** | ❌ **FT thua**: cùng kiểu lỗi |
| R14 | "Giải thích ngắn gọn quang hợp là gì." | "ánh sáng", "cây" | văn xuôi — 1.0 | `{"intent": "definition", "content": "…thực vật…"}` — 0.5 | ❌ **FT thua**: JSON bọc ngoài làm câu trả lời nghèo đi |
| T39 | "…nồi chiên… Hoàn tiền. Khi nào tiện. Quá tệ." | `urgency = thap` | `cao` — 0.75 | `trung_binh` — 0.75 | ➖ Cả hai sai `urgency` |
| R6 | "2 mũ 10 bằng bao nhiêu?" | có "1024" | giải từng bước, bị cắt ở 96 token — 0.0 | `{"answer": "1024", …}` — 1.0 | ✅ FT "thắng" — nhưng nhờ ngắn gọn, đây là nhiễu của chỉ số |

**Có mẫu chung ở các ca FT thua không?** Có, và có hai mẫu tách biệt:

1. **Trên regression**: mọi thất bại đều là *JSON hoá* câu hỏi không phải ticket. Đây là lỗi
   tổng quát hoá của định dạng, không phải mất kiến thức.
2. **Trên target**: cả **6/6** ticket bản fine-tune sai (T3, T5, T12, T39, T41, T46) đều chứa
   cụm **"Khi nào tiện"** và đều sai đúng một trường — `urgency` (nhãn `thap`, đoán
   `trung_binh`); ba trường còn lại đúng hết (6 × 0.25 / 50 = 0.03 → đúng bằng 1 − 0.97). Điều
   bất ngờ là trong tập train, "Khi nào tiện" đi với `thap` **30/30 lần** — một quy tắc hoàn
   toàn nhất quán mà model vẫn không học được sau 30 step, trong khi các cụm khác ("Gấp" → `cao`,
   "Hỏi cho biết thôi" → `thap`) thì học được. (b) cũng sai ở 6 ticket này (điểm 0.5–0.75),
   nên đây là điểm mù chung chứ không phải do fine-tune gây ra. Giả thuyết tôi chưa kiểm
   chứng: "Khi nào…" về ngữ nghĩa giống một câu hỏi thúc giục, nên tiên nghiệm của base
   (`trung_binh`) thắng 30 mẫu train.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy adapter này như một model đa dụng, nhưng lý do không phải
là nó kém ở tác vụ. Trên tác vụ triage, fine-tune thắng mốc (b) đã được prompt tử tế một khoảng
lớn và có thật (+0.205 target, format giữ 1.0, prompt ngắn hơn), và phần lớn lợi ích đến từ
việc phân biệt ý định tốt hơn (`doi_tra` vs `hoan_tien`, `van_chuyen` vs `hoi_thong_tin`) —
đúng loại kiến thức miền mà một ví dụ trong prompt không truyền tải hết. Nhưng cổng hồi quy
FAILED vì một tác dụng phụ có cơ chế rõ ràng: 225 mẫu đồng nhất "hỏi → JSON triage" đã dạy
model rằng mọi câu trả lời là JSON, kể cả "1 km bằng bao nhiêu mét". Nếu adapter chỉ phục vụ
một endpoint mà đầu vào luôn là ticket (chính là kịch bản hot-swap ở NB6), tác dụng phụ đó
không chạm tới người dùng và adapter đáng dùng; nếu nó thay thế model chung, nó gây hại. Bước
tiếp theo đúng là thêm 1–5 % dữ liệu replay phổ thông rồi chạy lại cổng, không nới ngưỡng.
Còn latency của (c) cao hơn (b) dù prompt ngắn hơn (1465.8 vs 1087.5 ms) — tôi cho là do
adapter chưa merge: `attn_only` gắn vào ít module hơn chỉ tốn 898.8 ms, và NB6 merge sẽ đưa
chi phí này về 0. Về đòn bẩy: trong lab này, **learning rate** là đòn bẩy lớn nhất (sai một
bậc → mất toàn bộ tác vụ), **mask** là điều kiện cần (đã chứng minh đúng), còn **vị trí
adapter** và **lượng tử hoá** chỉ tạo khác biệt 0 và 0.03 trên tập target đã gần bão hoà.
Giới hạn thật của kết quả giờ nằm ở **dữ liệu**: một quy tắc 30/30 nhất quán vẫn bị bỏ qua, và
tính đồng nhất của 225 mẫu chính là nguyên nhân của hồi quy.

**Ba điều tôi học được:**
1. `final_loss` là loss trung bình cả run — nó xếp `attn_only` đầu bảng, nhưng trên target hai
   run hoà (0.97). Xếp hạng phải bằng điểm tác vụ.
2. Regression tụt ≠ quên kiến thức: model vẫn biết đáp án, chỉ JSON hoá mọi câu trả lời. Chỉ
   đọc output mới thấy điều đó.
3. Trên Qwen3.5 lai, "chỉ q,v" chỉ chạm 8/32 lớp — phải in `layer_types` của chính model
   trước khi suy luận về vị trí adapter.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn ~5 % dữ liệu hỏi-đáp phổ thông vào train rồi
chạy lại NB3 + NB5 để xem cổng hồi quy có PASS mà target giữ ~0.97 không; (2) thêm vài mẫu
"Khi nào tiện" đối nghịch hoặc tăng epoch để kiểm tra giả thuyết về điểm mù `urgency`;
(3) chạy NB6 để xác nhận merge xoá chênh lệch latency.

---

## Phụ lục

**Artefact bổ sung (không thuộc pipeline chấm, không sửa file nào đã đóng băng):**
`scripts/qual_probe.py` → `results/qualitative_probe.json` — so sánh từng mẫu (b) vs (c) trên
10 ticket target và 15 câu regression, dùng đúng prompt và cách decode của NB2/NB5.

**Ghi chú quy trình:** NB1 được chạy thêm một lần trên máy local (M1, chỉ tokenizer) trước khi
lên Colab; `results/` nộp kèm là bản từ Colab (cùng commit), số liệu NB1 hai lần trùng khớp.

**Thưởng đã làm:**
- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
