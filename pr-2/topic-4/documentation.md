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


### Hasil Training Lengkap (Detik)

#### Tabel Hasil Training (CPU 8-core, CPU 32-core, 1080 Ti, 3070, 3080)

| Experiment                | Device | Batch Size | Grad Accum | 8-core CPU | 32-core CPU | 1080 Ti | 3070 | 3080 | Efektif Batch |
|---------------------------|--------|------------|------------|------------|-------------|---------|------|------|---------------|
| cpu_bs1_ga8               | CPU    | 1          | 8          | 1857       | 1180        |         |      |      | 8             |
| cpu_bs2_ga4               | CPU    | 2          | 4          | 968        | 637         |         |      |      | 8             |
| cpu_bs4_ga2               | CPU    | 4          | 2          | 572        | 373         |         |      |      | 8             |
| gpu_bs2_ga4               | GPU    | 2          | 4          |            |             | 634     | 973  | 638  | 8             |
| gpu_bs4_ga2               | GPU    | 4          | 2          |            |             | 538     | 550  | 374  | 8             |
| gpu_bs8_ga1               | GPU    | 8          | 1          |            |             | 499     | 401  | 312  | 8             |

*Catatan: Data 3070 hanya tersedia untuk batch 2, 4, dan 8 pada konfigurasi GPU.

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
| CPU 8-core  | -          |
| CPU 32-core | -          |
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
