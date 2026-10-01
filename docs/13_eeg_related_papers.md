# Paper EEG liên quan đến DriverGuard

Ngày tra cứu: 01/10/2026. Mục tiêu: model camera nhận diện dấu hiệu buồn ngủ,
sau đó EEG trong lab giám sát khả năng cảnh báo sớm; khi triển khai chỉ dùng
camera. Danh mục này ưu tiên nguồn gốc, không coi nội dung minh họa trong video
tham chiếu là dữ liệu thực nghiệm.

Metadata được đối chiếu qua Crossref, PubMed/Europe PMC và trang tác giả/dataset.
Ghi rõ phạm vi đã đọc: toàn văn với DROZY, Cao, Lin, ID3RSNet; abstract hoặc
trang chính thức với các nguồn khác, trừ khi có ghi chú riêng. Accuracy của
các paper EEG không thể so sánh trực tiếp với camera v1 vì khác dữ liệu,
nhãn, cohort và protocol. DOI có năm khác năm xuất bản ở một số bài.

## Các bài nên đọc trước

### 1. DROZY: cặp video NIR và EEG

**Massoz, Q., Langohr, T., François, C., & Verly, J. G. (2016).**
*The ULg multimodality drowsiness database (called DROZY) and examples of use.*
IEEE Winter Conference on Applications of Computer Vision (WACV).
[DOI: 10.1109/WACV.2016.7477715](https://doi.org/10.1109/WACV.2016.7477715).

Nguồn: [dataset chính thức](https://www.drozy.ulg.ac.be/),
[ORBi và dữ liệu](https://orbi.uliege.be/handle/2268/191620),
[toàn văn](https://orbi.uliege.be/bitstream/2268/191620/1/2016_WACV-copyrighted.pdf).

- 14 người; lab PVT, không phải lái xe trên đường. Bài gốc xác nhận ghi đồng
  bộ thời gian: video mặt NIR và PSG có năm kênh EEG Fz/Pz/Cz/C3/C4, 512 Hz.
  Có thêm EOG, ECG, EMG, reaction time và nhãn KSS.
- Mỗi PVT dài 10 phút; KSS tự báo cáo ngay trước test. Các PVT 2/3 tắt đèn.
  Đây là nguồn phù hợp nhất trong shortlist để thử camera–EEG và điều kiện tối.
- Bản công bố hiện có `videos_i8`, timestamp theo frame, EDF, KSS và PVT-RT.
  Raw NIR/depth bị bỏ khỏi bản tải từ 07/2018; vẫn giữ video NIR mã hóa.
  Một số phiên mất dữ liệu/frame; một số video 15 FPS do lỗi ghi khi tối.
  Phải sử dụng timestamp và kiểm tra `interpIndices`, không suy ra mọi phiên 30 FPS.
- **Chưa có nhãn biological sleep onset theo thời gian được xác minh.**
  Có EEG thô không đồng nghĩa đã có đáp án onset. KSS theo phiên không đủ
  làm nhãn cảnh báo sớm từng giây; cần protocol tạo/xác nhận nhãn EEG.
- [Giấy phép chính thức](https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf)
  giới hạn nghiên cứu/đánh giá, không phát triển thương mại, không phân phối
  lại hay sửa database; ảnh chỉ được công bố khi được đánh dấu cho phép.
  Giữ dữ liệu gốc và tuân thủ giấy phép khi thiết kế pipeline/đưa vào luận văn.

### 2. CNN EEG một kênh, có giải thích

**Cui, J., et al. (2022; online 2021).**
*A compact and interpretable convolutional neural network for cross-subject
driver drowsiness detection from single-channel EEG.* Methods, 202, 173–184.
[DOI: 10.1016/j.ymeth.2021.04.017](https://doi.org/10.1016/j.ymeth.2021.04.017).
[PubMed/abstract](https://pubmed.ncbi.nlm.nih.gov/33901644/).

CNN nhỏ, Global Average Pooling và Class Activation Map; abstract báo accuracy
trung bình 73,22% trên 11 người với phân loại cross-subject. Alpha spindle và
theta burst góp phần nhận diện buồn ngủ; mô hình cũng dùng muscle artifact
và sensor drift để nhận diện tỉnh. Đây là cảnh báo thực nghiệm về shortcut
learning: phải kiểm soát nhiễu trước khi dùng model EEG làm teacher.

Phù hợp làm tham chiếu kiến trúc teacher gọn và giải thích đặc trưng. Bài
phân loại hiện trạng, chưa chứng minh camera-only hoặc lead time trước onset.
Đã đọc abstract; chưa xác minh chi tiết validation bằng toàn văn.

### 3. CNN EEG cross-subject đa chiều

**Cui, J., Lan, Z., Sourina, O., & Müller-Wittig, W. (2023; online 2022).**
*EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable
Convolutional Neural Network.* IEEE Transactions on Neural Networks and
Learning Systems, 34, 7921–7933.
[DOI: 10.1109/TNNLS.2022.3147208](https://doi.org/10.1109/TNNLS.2022.3147208).
[PubMed/abstract](https://pubmed.ncbi.nlm.nih.gov/35171778/).

Separable convolution học đặc trưng không gian/thời gian và diễn giải theo
mẫu; abstract báo leave-one-out cross-subject trên 11 người, accuracy 78,35%.
Đây là bài khác với Methods ở mục 2; không gán lẫn kết quả hoặc kiến trúc.
Hữu ích cho lựa chọn teacher EEG và đánh giá trên người chưa thấy; chưa có
bằng chứng EEG dạy camera cảnh báo trước onset trong nguồn đã đọc.

### 4. EEG và hành vi lái xe có timestamp

**Cao, Z., Chuang, C.-H., King, J.-K., & Lin, C.-T. (2019).**
*Multi-channel EEG recordings during a sustained-attention driving task.*
Scientific Data, 6, 19.
[DOI: 10.1038/s41597-019-0027-4](https://doi.org/10.1038/s41597-019-0027-4).
[Toàn văn PMC](https://europepmc.org/articles/PMC6472414).

27 người, 62 phiên, mô phỏng VR highway ban đêm, phiên 90 phút. Ghi 30 điện
cực EEG cùng hai reference, 500 Hz, vị trí xe và các sự kiện lệch làn/phản ứng
sửa làn. Reaction time là tham chiếu vigilance, không phải nhãn EEG sleep onset.
Các event deviation onset/response onset/offset không được đổi tên thành
thời điểm bắt đầu buồn ngủ.

Phù hợp nghiên cứu/pretrain teacher EEG có liên hệ hành vi. Dashboard camera
có quan sát mặt nhưng phần dữ liệu công bố được xác minh gồm EEG/xe/events,
chưa xác minh video mặt được phát hành. Không dùng bộ này ghép với UTA thành
cặp EEG–camera vì không cùng người/cùng phiên.

### 5. Teacher EEG gọn với residual shrinkage

**Feng, X., Guo, Z., & Kwong, S. (công bố 08/01/2025; volume 2024).**
*ID3RSNet: cross-subject driver drowsiness detection from raw single-channel
EEG with an interpretable residual shrinkage network.* Frontiers in Neuroscience, 18.
[DOI: 10.3389/fnins.2024.1508747](https://doi.org/10.3389/fnins.2024.1508747).
[Toàn văn PMC](https://europepmc.org/articles/PMC11751225).

Residual shrinkage, attention, soft threshold, GAP và ECAM; dùng Oz và
leave-one-subject-out trên 11 người từ dữ liệu SADT/Cao. Mẫu EEG dài ba giây
trước sự kiện lệch làn, nhãn dựa trên reaction time. Cách chọn epoch/ngưỡng
của DriverGuard vẫn cần validation riêng theo người; không sao chép cách báo
peak epoch như một protocol test đã đóng băng.

Hữu ích cho denoising và baseline teacher raw EEG. **Ba giây trước lệch làn
không có nghĩa đã dự báo trước biological sleep onset ba giây.** Bài không
chứng minh student camera-only hoặc hiệu quả NIR.

### 6. SEED-VIG: EEG và EOG, nhãn từ mắt

**Zheng, W.-L., & Lu, B.-L. (2017).**
*A multimodal approach to estimating vigilance using EEG and forehead EOG.*
Journal of Neural Engineering, 14(2), 026017.
[DOI: 10.1088/1741-2552/aa5a98](https://doi.org/10.1088/1741-2552/aa5a98).
[Dataset chính thức](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html),
[thủ tục truy cập](https://bcmi.sjtu.edu.cn/home/seed/downloads.html).

Fusion EEG + forehead EOG, mô hình phụ thuộc thời gian; nhãn vigilance là
PERCLOS từ eye-tracking glasses. Trang release liệt kê EEG PSD/DE, forehead
PSD/DE, EOG feature và nhãn PERCLOS; chưa xác minh có video mặt, eye-tracker
stream hoặc raw EEG trong các phần tải được mô tả. Truy cập qua đăng ký/license.

Phù hợp nghiên cứu vigilance liên tục và fusion, chưa đủ làm bộ cặp
camera–EEG cho distillation. **EOG là điện nhãn đồ, không phải camera mắt.**
Teacher được train theo PERCLOS không tự cung cấp một đáp án sinh lý độc lập
với dấu hiệu nhắm mắt để chứng minh cảnh báo sớm.

### 7. Nền tảng EEG và sai lệch làn đường

**Lin, C.-T., et al. (2005).**
*EEG-Based Drowsiness Estimation for Safety Driving Using Independent
Component Analysis.* IEEE Transactions on Circuits and Systems I: Regular
Papers, 52(12), 2726–2738.
[DOI: 10.1109/TCSI.2005.857555](https://doi.org/10.1109/TCSI.2005.857555).
[PDF tác giả](https://sccn.ucsd.edu/~jung/pdf/IEEECAS05.pdf).

ICA + log-power spectra + linear regression ước lượng sai lệch làn. Thu 16
người, chọn năm người/10 phiên có microsleep để modeling/test giữa các phiên;
target làm trơn cửa sổ nhân quả 90 giây, bước hai giây. Đây là nền tảng EEG
vigilance nhưng đánh giá theo người qua phiên, chưa phải tổng quát người mới.
Không trích dẫn như bằng chứng camera cảnh báo trước onset.

### 8. Alpha spindle trong lái xe đường thật

**Simon, M., et al. (2011).**
*EEG alpha spindle measures as indicators of driver fatigue under real traffic
conditions.* Clinical Neurophysiology.
[DOI: 10.1016/j.clinph.2010.10.044](https://doi.org/10.1016/j.clinph.2010.10.044).
[PubMed/abstract](https://pubmed.ncbi.nlm.nih.gov/21333592/).

So sánh 20 phút đầu với 20 phút cuối trước khi người tham gia dừng vì mệt;
alpha spindle metrics tốt hơn alpha-band power về các chỉ số tác giả xét.
Hữu ích cho cơ sở sinh lý ngoài lab và lý do nghiên cứu cấu trúc burst thay
vì chỉ công suất trung bình. Abstract chưa đủ xác minh cohort/split chi tiết;
không chứng minh horizon cảnh báo sớm từng giây hay distillation.

### 9. Microsleep được ghi cùng eye-video và EEG

**Poudel, G. R., Innes, C. R., Bones, P. J., Watts, R., & Jones, R. D. (2014).**
*Losing the struggle to stay awake: divergent thalamic and cortical activity
during microsleeps.* Human Brain Mapping.
[DOI: 10.1002/hbm.22178](https://doi.org/10.1002/hbm.22178).
[PMC/abstract](https://europepmc.org/articles/PMC6869765).

20 người làm continuous tracking 50 phút, ghi đồng thời EEG, eye-video, fMRI
và phản hồi. Microsleep được mô tả là mất phản hồi ngắn 0,5–15 giây kèm
nhắm mắt chậm; đánh giá bằng hành vi và eye-video. Phù hợp thiết kế annotation
sự kiện và phân biệt drowsiness với microsleep. Đây là lab tracking, chưa
phải cảnh báo sớm khi lái xe; chưa xác minh dataset video công khai.

### 10. Kiểm chứng nhãn KSS với EEG và hành vi

**Kaida, K., et al. (2006).**
*Validation of the Karolinska sleepiness scale against performance and EEG
variables.* Clinical Neurophysiology.
[DOI: 10.1016/j.clinph.2006.03.011](https://doi.org/10.1016/j.clinph.2006.03.011).

16 phụ nữ khỏe mạnh, lặp đo PVT, KSS, EEG alpha/theta và alpha attenuation
test trong ba ngày. Abstract cho thấy KSS liên quan EEG/hành vi. Hỗ trợ
protocol thu nhãn chủ quan cùng EEG; không biến một KSS theo phiên thành
nhãn onset chính xác cho mỗi frame hoặc bảo đảm tổng quát mọi tài xế.

## Tổng quan và phương pháp bổ trợ

| Nguồn đã xác minh | Vai trò và giới hạn |
|---|---|
| Li & Chung (2022), *Electroencephalogram-Based Approaches for Driver Drowsiness Detection and Management: A Review*. [DOI 10.3390/s22031100](https://doi.org/10.3390/s22031100); [PMC](https://europepmc.org/articles/PMC8840041) | Tổng quan EEG DDD, detection và management; dùng mở đầu literature review, không thay bằng chứng thí nghiệm riêng. |
| Stancin, Cifrek & Jovic (2021), *A Review of EEG Signal Features and their Application in Driver Drowsiness Detection Systems*. [DOI 10.3390/s21113786](https://doi.org/10.3390/s21113786); [PMC](https://europepmc.org/articles/PMC8198610) | Tổng quan feature EEG và deep learning; chọn feature/ablation, không suy ra một dải tần là đáp án onset. |
| Gupta, Hoffman & Malik (2016), *Cross Modal Distillation for Supervision Transfer*. [DOI 10.1109/CVPR.2016.309](https://doi.org/10.1109/CVPR.2016.309); [CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Gupta_Cross_Modal_Distillation_CVPR_2016_paper.html) | Representation transfer RGB sang depth/optical flow bằng dữ liệu ghép cặp. Cơ sở phương pháp; không EEG và không buồn ngủ. |
| Vapnik & Vashist (2009), *A new learning paradigm: Learning using privileged information*. [DOI 10.1016/j.neunet.2009.06.042](https://doi.org/10.1016/j.neunet.2009.06.042) | LUPI: thông tin phụ dùng khi train. Hợp lý luận EEG chỉ trong lab; chưa chứng minh camera cảnh báo sớm. |
| Lu, Huang, Yi & Pan (2025), *Confidence-Aware Multimodal Network for Driver Drowsiness Detection: Leveraging EEG-EOG Signals Enhanced by Knowledge Distillation*. [DOI 10.1109/TIM.2025.3591857](https://doi.org/10.1109/TIM.2025.3591857) | Metadata và abstract indexed đã kiểm tra; confidence-aware EEG/EOG fusion và KD. Chưa đọc full text; abstract không xác định student nhận modality nào, dataset/split hoặc camera. Không gọi đây là EEG teacher dạy camera. |
| Caffier, Erdmann & Ullsperger (2003), *Experimental evaluation of eye-blink parameters as a drowsiness measure*. [DOI 10.1007/s00421-003-0807-5](https://doi.org/10.1007/s00421-003-0807-5) | 60 người, cảm biến IR trên kính, blink duration/reopening; abstract dùng bảng hỏi. Hỗ trợ feature camera, chưa chứng minh EEG onset. |
| Schleicher, Galley, Briest & Galley (2008), *Blinks and saccades as indicators of fatigue in sleepiness warnings: looking tired?* [DOI 10.1080/00140130701817062](https://doi.org/10.1080/00140130701817062) | 129 người, blink/reopening/saccade và khác biệt giữa người. Hỗ trợ temporal features và subject split; chưa chứng minh distillation EEG–camera. |

Các nguồn bổ trợ được đọc ở mức abstract/trang công bố, ngoại trừ trang CVF
của Gupta. Với Lu, abstract có qua index OpenAlex; IEEE không trả toàn văn
trong lần truy cập này, nên không dùng accuracy của bài làm benchmark dự án.

Hai bài gần hơn về **EEG kết hợp video**, nhưng chưa chứng minh EEG chỉ dùng
khi train rồi bỏ ở inference:

- **Shepelev, D., & Demyanenko, Y. (2025)**, *A Neural Network Committee for
  Multimodal EEG and Video Drowsiness Detection*.
  [DOI 10.1007/978-3-032-07690-8_23](https://doi.org/10.1007/978-3-032-07690-8_23).
  Publisher abstract mô tả năm người, fusion classifier trên blink artifact
  Fp1/Fp2, alpha O1/O2 và trạng thái mắt từ face video. Full text cần truy cập;
  chưa xác minh public dataset, timestamp, onset hay subject-independent split.
- **Shanthini, S., & Subhashini, S. (2024)**, *Multimodal Fusion for Robust
  Driver Drowsiness Detection: Integrating EEG and Video Signals with Enhanced Accuracy*.
  [DOI 10.1109/ACCAI61061.2024.10601871](https://doi.org/10.1109/ACCAI61061.2024.10601871).
  Metadata Crossref và abstract indexed OpenAlex mô tả EEG SVM + multiscale
  CNN video fusion; publisher không trả toàn văn trong lần truy cập. Chưa
  xác minh cohort, dữ liệu đồng bộ, split, onset hoặc student camera-only.

Nguồn camera mới liên quan bước kiểm chứng IR nhưng **không có EEG**:
**Bodaghi, M., et al. (2026)**, *A multimodal drowsiness dataset using video,
biometric, and behavioral data*, Scientific Data (UL-DD).
[DOI 10.1038/s41597-025-06540-1](https://doi.org/10.1038/s41597-025-06540-1),
[toàn văn](https://europepmc.org/articles/PMC13039290).
19 người, video RGB/depth/IR, nhãn KSS bốn phút/lần và biometrics khác; không
thay thế cặp EEG–camera để teacher supervision.
[Zenodo](https://zenodo.org/records/17978727) hiện yêu cầu truy cập: API trả
`access_right=restricted`, chưa liệt kê file tải công khai, dù bài gọi dataset
là publicly available. Không mặc định có thể tải toàn bộ ngay.

## Kết luận áp dụng cho đề tài

1. Ưu tiên kiểm tra DROZY cho thử nghiệm dữ liệu ghép cặp và NIR. Bản tải
   chưa được lấy hoặc kiểm tra file trong phiên tìm paper này; cần xác minh
   alignment EDF/video, missing frames, nhãn và khả năng landmark NIR trước train.
2. Dùng Cao/Cui/ID3RSNet làm tham chiếu hoặc pretrain teacher EEG, nhưng phải
   xử lý khác biệt kênh/thiết bị/cohort với DROZY. Không ghép EEG của Cao và
   video UTA không đồng bộ thành cặp train.
3. Định nghĩa rõ mục tiêu: mức drowsiness hiện tại, microsleep hoặc nguy cơ
   onset tương lai. Nhãn RT/KSS/PERCLOS hỗ trợ từng mục tiêu khác nhau; EEG
   còn cần kiểm soát nhiễu và annotation trước khi làm tham chiếu onset.
4. Camera chỉ nhận hiện tại/quá khứ. EEG tương lai có thể tạo target forecasting
   ở train, nhưng không vào input student; validation/test tách người, giữ
   nhãn không đủ future horizon là unavailable. Báo sensitivity, lead time
   và false alerts/giờ, kể cả cảnh báo sau onset và trường hợp không đánh giá được.
5. So sánh camera gốc, camera thêm blink kinetics và camera có EEG supervision
   tại cùng false alerts/giờ, trên RGB/NIR và người chưa thấy. Đây là đóng
   góp dự kiến cần chứng minh; shortlist hiện tại chưa xác minh bài làm đúng
   toàn bộ luồng EEG chỉ khi train → camera-only cảnh báo sớm. Điều đó không
   phải kết luận rằng chưa ai nghiên cứu hướng này.

Thứ tự đọc gợi ý: **review Li/Chung → DROZY → Cao → Cui Methods → Cui TNNLS
→ ID3RSNet → Poudel/Kaida → Gupta/LUPI**. Kế hoạch triển khai hiện tại:
[10_camera_eeg_night_plan.md](10_camera_eeg_night_plan.md).
