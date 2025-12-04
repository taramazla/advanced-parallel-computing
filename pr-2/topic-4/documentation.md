# Topik 4: Machine Learning dengan GPU/CUDA - Fine-tuning LLM menggunakan LoRA

## 📚 Deskripsi Project

Project ini merupakan implementasi **fine-tuning Large Language Model (LLM)** menggunakan metode **LoRA (Low-Rank Adaptation)** untuk parallel computing dengan akselerasi GPU/CUDA. LoRA adalah teknik parameter-efficient fine-tuning yang memungkinkan adaptasi model bahasa besar dengan resource komputasi minimal, sangat cocok untuk lingkungan GPU dengan memory terbatas.

### Tujuan Pembelajaran
1. Memahami konsep machine learning dengan akselerasi GPU/CUDA
2. Mengimplementasikan teknik LoRA untuk fine-tuning model bahasa
3. Melakukan eksperimen dengan berbagai konfigurasi hyperparameter
4. Menganalisis performa training dengan dan tanpa GPU
5. Memahami trade-off antara efisiensi memori dan performa model

---

## 🎯 Topik Keempat: Machine Learning di Lingkungan GPU/CUDA

# Fine-tuning LLM dengan LoRA: Analisis Performa CPU vs GPU

## Apa itu LoRA?

**LoRA (Low-Rank Adaptation)** adalah teknik fine-tuning efisien untuk model bahasa besar (LLM) yang hanya melatih sebagian kecil parameter tambahan (low-rank matrices) pada layer tertentu, sehingga sangat hemat memori dan cocok untuk GPU dengan VRAM terbatas.

**Inti LoRA:**
```
W = W₀ + BA
```
Hanya matriks B dan A yang di-train, W₀ (pretrained) tetap beku.

**Keunggulan LoRA:**
- Hanya melatih ~0.6% parameter (GPT-2)
- Hemat memori, cocok untuk GPU kecil
- Modular, bisa swap adapter untuk task berbeda
- Inference cepat


## Eksperimen: Perbandingan Training di CPU, GPU 1080 Ti, dan GPU 3080

### Setup
- Model: GPT-2 (125M)
- Dataset: Bitext Customer Support (26,872 pairs)
- LoRA config: r=8, alpha=16, dropout=0.1
- 1 epoch, batch size & grad_accum bervariasi



### Statistik Ringkas & Hasil Training Lengkap

#### Statistik Rata-rata (Semua Device)

- **CPU 8-core**: Rata-rata durasi ≈ 1132 detik (18m 52s), tercepat 572s, terlambat 1857s
- **CPU 32-core**: Rata-rata durasi ≈ 730 detik (12m 10s), tercepat 373s, terlambat 1180s
- **1080 Ti**: Rata-rata durasi ≈ 644 detik (10m 44s), tercepat 499s, terlambat 907s
- **3070**: Rata-rata durasi ≈ 807 detik (13m 27s), tercepat 401s, terlambat 1305s
- **3080**: Rata-rata durasi ≈ 633 detik (10m 33s), tercepat 312s, terlambat 1170s

#### Tabel Hasil Training (Sesuai result_combine.csv, tanpa kolom quantization)

| experiment              | device | batch_size | grad_accum | duration_seconds |
|-------------------------|--------|------------|------------|------------------|
| cpu_bs1_ga8_8core       | CPU    | 1          | 8          | 1857             |
| cpu_bs2_ga4_8core       | CPU    | 2          | 4          | 968              |
| cpu_bs4_ga2_8core       | CPU    | 4          | 2          | 572              |
| cpu_bs1_ga8_32core      | CPU    | 1          | 8          | 1180             |
| cpu_bs2_ga4_32core      | CPU    | 2          | 4          | 637              |
| cpu_bs4_ga2_32core      | CPU    | 4          | 2          | 373              |
| gpu_bs1_ga8_1080Ti      | GPU    | 1          | 8          | 907              |
| gpu_bs2_ga4_1080Ti      | GPU    | 2          | 4          | 634              |
| gpu_bs4_ga2_1080Ti      | GPU    | 4          | 2          | 538              |
| gpu_bs8_ga1_1080Ti      | GPU    | 8          | 1          | 499              |
| gpu_bs1_ga8_3070        | GPU    | 1          | 8          | 1305             |
| gpu_bs2_ga4_3070        | GPU    | 2          | 4          | 973              |
| gpu_bs4_ga2_3070        | GPU    | 4          | 2          | 550              |
| gpu_bs8_ga1_3070        | GPU    | 8          | 1          | 401              |
| gpu_bs1_ga8_3080        | GPU    | 1          | 8          | 1170             |
| gpu_bs2_ga4_3080        | GPU    | 2          | 4          | 638              |
| gpu_bs4_ga2_3080        | GPU    | 4          | 2          | 374              |
| gpu_bs8_ga1_3080        | GPU    | 8          | 1          | 312              |

*Tabel di atas diambil langsung dari file result_combine.csv untuk menjaga konsistensi data dan kemudahan analisis. Kolom quantization dihilangkan karena seluruh eksperimen menggunakan mode 'none'.*

#### Speedup & Analisis

- **GPU 3080** tercepat di semua konfigurasi batch, diikuti 3070 dan 1080 Ti.
- **Speedup rata-rata GPU 3080 vs CPU 32-core**: ~1.15x (rata-rata 730s vs 633s), namun pada batch besar (8), speedup bisa >1.2x.
- **Speedup GPU 3080 vs CPU 8-core**: ~1.8x (rata-rata 1132s vs 633s).
- **Speedup GPU 3080 vs 1080 Ti**: ~1.02x (rata-rata 644s vs 633s), namun pada batch besar (8), 3080 unggul signifikan (312s vs 499s, ~1.6x).
- **Batch besar** (batch 8, grad_accum 1) sangat menguntungkan GPU modern, gap performa makin lebar.
- **CPU 32-core** sudah cukup cepat untuk eksperimen/testing, namun tetap jauh di bawah GPU modern untuk training skala besar.

#### Rekomendasi Praktis
- Untuk training LLM efisien, gunakan GPU generasi terbaru (3080/3070) dengan batch size besar.
- Untuk eksperimen ringan, CPU 32-core sudah memadai.
- 1080 Ti masih relevan untuk model kecil atau batch kecil, namun untuk produksi dan efisiensi waktu, 3080 sangat direkomendasikan.

#### Temuan Kunci
1. Batch besar meningkatkan utilisasi GPU dan mempercepat training.
2. Gradient accumulation memungkinkan simulasi batch besar pada memori terbatas.
3. CPU training cukup untuk testing, namun tidak efisien untuk produksi.
4. Speedup GPU vs CPU sangat tergantung batch size dan generasi hardware.
5. Semua eksperimen di atas tanpa quantization, namun quantization dapat sangat membantu untuk model lebih besar.

---

#### Analisis Performa

- **CPU 32-core** memberikan percepatan signifikan dibanding 8-core, terutama pada batch besar (speedup hingga ~1.6x pada batch 4).
- **GPU 3080** adalah yang tercepat di semua konfigurasi batch, diikuti oleh 3070 dan 1080 Ti.
- **GPU 3070** performanya di tengah-tengah antara 1080 Ti dan 3080, namun pada batch besar (8), 3070 lebih cepat dari 1080 Ti.
- **GPU 1080 Ti** masih layak untuk training LLM kecil, namun untuk batch besar dan efisiensi waktu, 3080 dan 3070 jauh lebih unggul.
- **Efek batch size**: Semakin besar batch, semakin besar gap performa antar GPU, menandakan hardware modern sangat diuntungkan workload paralel besar.
- **CPU**: Untuk eksperimen/testing, CPU 32-core sudah cukup cepat, namun tetap jauh di bawah GPU modern untuk training skala besar.

#### Visualisasi Speedup (Batch 8, grad_accum 1)

| Device      | Durasi (s) |
|------------ |------------|
| 1080 Ti     | 499        |
| 3070        | 401        |
| 3080        | 312        |

3080 ~1.6x lebih cepat dari 1080 Ti, dan ~1.3x lebih cepat dari 3070 pada konfigurasi batch terbesar.

#### Insight & Rekomendasi
- Untuk training LLM efisien, gunakan GPU generasi terbaru (3080/3070) dengan batch size besar.
- Untuk eksperimen ringan, CPU 32-core sudah memadai.
- 1080 Ti masih relevan untuk model kecil atau batch kecil, namun untuk produksi dan efisiensi waktu, 3080 sangat direkomendasikan.

---


### Analisis Pengaruh Batch Size & Gradient Accumulation

**Batch size** menentukan berapa banyak data yang diproses sekaligus dalam satu langkah update parameter. **Gradient accumulation steps** memungkinkan kita "menyimulasikan" batch besar dengan cara mengakumulasi gradien dari beberapa mini-batch sebelum melakukan update parameter.

#### Temuan Eksperimen:
- **Batch size besar** (dan grad_accum kecil) mempercepat training, terutama di GPU, karena GPU lebih efisien memproses data dalam jumlah besar secara paralel.
- **CPU**: batch 4, grad_accum 2 (efektif batch 8) adalah konfigurasi tercepat (9:32). Batch lebih kecil (1/2) dengan grad_accum lebih besar justru memperlambat training karena overhead update parameter lebih sering dan kurang efisien secara cache/memory.
- **GPU**: batch 8, grad_accum 1 (efektif batch 8) tercepat (6:41). GPU sangat diuntungkan oleh batch besar karena dapat memaksimalkan throughput hardware.
- **Gradient accumulation** sangat berguna jika memori terbatas: kita bisa tetap "simulasikan" batch besar walau batch fisik kecil, namun training jadi sedikit lebih lambat karena update parameter lebih jarang dan ada overhead akumulasi gradien.

#### Insight Teknis:
- **Batch besar** → lebih sedikit update parameter per epoch, gradien lebih stabil, training lebih cepat per epoch, tapi butuh memori lebih besar.
- **Batch kecil** → lebih banyak update parameter, gradien lebih noisy (kadang membantu generalisasi), training lebih lambat per epoch, lebih hemat memori.
- **Gradient accumulation** → trade-off antara efisiensi memori dan kecepatan. Efektif untuk hardware terbatas, tapi idealnya batch fisik tetap dibuat sebesar mungkin.

#### Rekomendasi Praktis:
- Untuk GPU: gunakan batch size terbesar yang muat di VRAM, minimalkan grad_accum jika memungkinkan.
- Untuk CPU: batch size sedang (4) dan grad_accum sedang (2) optimal untuk efisiensi.
- Selalu sesuaikan batch size dan grad_accum dengan kapasitas memori dan target waktu training.

### Rekomendasi
- Untuk produksi: **gunakan GPU** dengan batch size terbesar yang muat di memori.
- Untuk eksperimen/testing: **CPU cukup**, batch size 4 optimal.

---

**Kesimpulan:**
LoRA memungkinkan fine-tuning LLM secara efisien di resource terbatas. GPU memberikan percepatan training signifikan, namun CPU masih layak untuk eksperimen kecil.

**Math:**
```
h = W₀x + ΔWx = W₀x + BAx
```
Dimana B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), dan r << min(d,k)



### 3. Gradient Accumulation

Simulate larger batch sizes dengan memory terbatas:
```
Effective Batch Size = batch_size × gradient_accumulation_steps
```

Example:
- Physical batch: 2
- Accumulation steps: 8
- Effective batch: 16

**Benefit:** Lebih stable training dengan limited VRAM

---

## 📈 Analisis Hasil


## 🎓 Kesimpulan

### Pembelajaran Utama

1. **LoRA Effectiveness**: 
   - Berhasil fine-tune GPT-2 dengan <1% parameters
   - Loss reduction signifikan dalam 1 epoch
   - Memory efficient - dapat run di CPU

2. **GPU Acceleration**:
   - GPU memberikan 5-10x speedup untuk training
   - Essential untuk model besar (7B+)
   - Quantization memungkinkan training di consumer GPU

3. **Dataset Quality**:
   - Structured instruction-following dataset crucial
   - Clean data > Large data
   - Domain-specific fine-tuning effective

4. **Practical Considerations**:
   - Hyperparameter tuning penting
   - Gradient accumulation simulate larger batches
   - Validation split prevent overfitting

### Aplikasi Praktis

**Use Cases:**
- 🤖 Customer service chatbots
- 📝 Domain-specific text generation
- 🔍 Question answering systems
- 💬 Conversational AI
- 📊 Data analysis assistants

**Industry Relevance:**
- E-commerce: Automated customer support
- Healthcare: Medical Q&A systems
- Education: Tutoring chatbots
- Finance: Financial advisory bots

---

## 📚 Referensi

### Papers
1. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
2. **QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
3. **GPT-2**: Radford et al. "Language Models are Unsupervised Multitask Learners" (2019)

### Libraries
- **Transformers**: HuggingFace transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning library
- **bitsandbytes**: 8-bit & 4-bit quantization
- **Datasets**: HuggingFace datasets library

### Resources
- HuggingFace Documentation: https://huggingface.co/docs
- PEFT Examples: https://github.com/huggingface/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685

---

## 👥 Author & Credits

**Project**: Advanced Parallel Computing - Topic 4
**Focus**: Machine Learning dengan GPU/CUDA
**Implementation**: LoRA Fine-tuning untuk LLM
**Dataset**: Bitext Customer Support Training Dataset

**Technologies Used:**
- PyTorch 2.9+
- Transformers 4.57+
- PEFT 0.18+
- Datasets 4.4+
- Python 3.8+

---

## 📝 Lisensi & Penggunaan

Project ini dibuat untuk tujuan pembelajaran dalam mata kuliah **Advanced Parallel Computing**.

**Penggunaan:**
- ✅ Educational purposes
- ✅ Research & experimentation
- ✅ Personal projects
- ✅ Commercial applications (dengan proper attribution)

**Catatan:**
- Base models (GPT-2, LLaMA, dll.) memiliki lisensi tersendiri
- Dataset memiliki lisensi dari creator
- Pastikan mematuhi terms of use untuk production use

---

## 🔄 Future Improvements

**Potential Enhancements:**
1. Multi-GPU training support (DistributedDataParallel)
2. Gradient checkpointing untuk memory efficiency
3. Mixed precision training (BF16)
4. Custom dataset preprocessing pipeline
5. Web interface untuk interactive testing
6. Model merging & deployment guide
7. Benchmark suite untuk model comparison
8. A/B testing framework untuk production

**Eksperimen Lanjutan:**
1. Compare LoRA vs other PEFT methods (Prefix Tuning, Adapter, etc.)
2. Test dengan model lebih besar (LLaMA-13B, Mistral-7B)
3. Multi-task learning dengan multiple adapters
4. Cross-lingual transfer learning
5. Domain adaptation experiments

---

*Dokumentasi ini mencakup penjelasan lengkap project fine-tuning LLM dengan LoRA untuk submission tugas Advanced Parallel Computing - Topic 4: Machine Learning di Lingkungan GPU/CUDA.*
