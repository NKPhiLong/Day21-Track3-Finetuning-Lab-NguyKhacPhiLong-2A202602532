# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Cụm "Khi nào tiện" đi với `urgency = thap` 30/30 lần trong tập train, vậy mà bản fine-tune sai
cả 6/6 ticket có cụm này ở tập eval — và đó là *toàn bộ* lỗi của nó trên target. Một quy tắc
hoàn toàn nhất quán vẫn không học được sau 30 step, trong khi "Gấp" → `cao` thì học ngay.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở train. Cả pipeline NB1→NB5 chỉ hết ~51 phút (README ước 100–130). Phần tốn thời
gian nhất là sau NB5: hiểu *vì sao* regression tụt 0.113. NB5 không lưu câu trả lời regression,
nên tôi phải viết thêm `scripts/qual_probe.py` để sinh lại (b) và (c) cạnh nhau rồi đọc từng
output. Tôi đã dự đoán sẽ kẹt ở OOM hoặc train, không phải ở khâu đọc kết quả.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ "regression tụt" nghĩa là model quên kiến thức. Ở đây model vẫn biết Hà Nội là
thủ đô — nó chỉ bọc mọi câu trả lời vào JSON (14/15 câu phổ thông), và với "1 km bằng bao
nhiêu mét?" thì trả về nguyên schema triage, không có đáp án. Thứ bị mất là *hành vi*, không
phải kiến thức. Tôi cũng không còn tin train loss thấp hơn = cấu hình tốt hơn: `attn_only` có
loss thấp nhất nhưng chỉ hoà `correct` trên target.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng Claude Code để đọc tài liệu, chạy NB1 + test ở máy, điều khiển Colab T4 chạy NB2→NB5,
kéo `results/` về, viết script đối chiếu định tính và soạn report. Chỗ nó sai: (1) bấm nhầm
"+ Văn bản" thay vì "+ Mã" trên Colab nên lệnh rơi vào ô markdown; (2) gõ vào sai chỗ khi
trang tự cuộn, phải kiểm tra lại ô `STAGES`; (3) trong report ban đầu viết "15/15 câu trả lời
là JSON" — kiểm tra lại bằng script thì thực tế là 14/15. Bài học: số nào
trong report cũng phải được đối chiếu với file, kể cả số do AI viết.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Dựng và đóng băng tập eval trước — gồm cả tác vụ của khách *và* một tập hồi quy phản ánh những
gì model đang làm tốt cho họ — rồi đo baseline prompt tốt nhất có thể. Lab này cho thấy prompt
(b) đã đạt 0.765 mà không cần train; nếu chỉ cần chừng đó thì không nên fine-tune.
