# FYP2024
# Low-Resource Fake News Detection with FastText + XLNet Features

A fake news classification pipeline that combines **static word embeddings (FastText)** with **contextual embeddings (XLNet)** as features, then compares four lightweight machine learning classifiers. The goal is to get useful detection accuracy **without the compute cost of fine-tuning large transformer models like BERT**.


---

## Motivation

Fake news spreads faster than true news on social media and can erode public trust, affect the economy, and influence political outcomes, as seen during the COVID-19 pandemic and the 2016 US Presidential Election. Manual fact-checking can't keep up with that volume.

Prior work shows that deep learning models such as BERT reach the highest accuracy (around 98%), but they need powerful GPUs/TPUs, long training times and large storage. This project targets **organisations and individuals with limited resources** (with a focus on the healthcare/COVID-19 domain) and asks:

Can combining cheap, pre-trained feature extractors with simple classifiers give a reasonable balance between accuracy and resource usage?

The project was also motivated by a prior study where Word2Vec features scored only 75.67% accuracy, so a second aim was to see whether enriching word embeddings with contextual XLNet embeddings improves on that.

---

## Approach

```
Raw tweets/posts
      │
      ▼
Cleaning (HTML unescape, remove @mentions, URLs, punctuation, stop words, lowercase)
      │
      ├──────────────► FastText (wiki.simple.bin) ──► mean word vector
      │
      └──────────────► XLNet (xlnet-base-cased) ────► mean-pooled last hidden state
                                │
                                ▼
                    Concatenated feature vector
                                │
                                ▼
        Logistic Regression │ Naive Bayes │ Random Forest │ SVM
                                │
                                ▼
       Accuracy, precision, recall, F1 + time taken (per machine)
```

### Feature extraction

| Extractor | Model used | Why |
|---|---|---|
| **FastText** | `wiki.simple.bin` (Simple English Wikipedia vectors) | Builds vectors from subword units, so it handles misspellings, hashtags and other out-of-vocabulary words common in social media. Lightweight compared to Google News Word2Vec. |
| **XLNet** | `xlnet-base-cased` | Permutation language modelling captures bidirectional context and long-range dependencies. The base model balances performance and resource use. Inputs are truncated to 512 tokens. |

Each text gets a FastText vector (average of its word vectors) and an XLNet vector (mean of the final hidden states). The two are concatenated into one feature vector.

### Classifiers

| Model | Configuration |
|---|---|
| Logistic Regression | `solver='lbfgs'`, `max_iter=5000` |
| Multinomial Naive Bayes | Features scaled with `MinMaxScaler` (MNB can't take negative values) |
| Random Forest | `n_estimators=100`, `criterion='entropy'`, `class_weight='balanced_subsample'`, `random_state=42` |
| SVM | `kernel='linear'` |

---

## Dataset

**COVID-19 fake news dataset** (Kaggle), collected from social media platforms including Twitter, Facebook and Instagram.

| Split | File | Entries |
|---|---|---|
| Train | `Constraint_Train` | 6,420 |
| Test | `Constraint_Test` | 2,140 |

Labels are binary: real vs fake.

---

## Results

### Classifier performance (MacBook Air M2)

| Model | Precision | Recall | F1 | Accuracy | Time (s) |
|---|---|---|---|---|---|
| **Random Forest** | **0.8732** | **0.8740** | **0.8733** | **87.34%** | 12.43 |
| SVM | 0.8693 | 0.8700 | 0.8695 | 86.96% | 64.15 |
| Logistic Regression | 0.8684 | 0.8691 | 0.8686 | 86.87% | 15.92 |
| Naive Bayes | 0.7932 | 0.7938 | 0.7929 | 79.30% | 0.08 |

### Feature extraction time (MacBook Air M2)

| | FastText | XLNet |
|---|---|---|
| Train set (6,420) | 1.17 s | 292.20 s |
| Test set (2,140) | 2.23 s | 96.47 s |

XLNet extraction is by far the most expensive step in the pipeline.

### Across machines

The pipeline was run on three machines. Accuracy stayed almost the same; run time did not.

| Model time (s) | MacBook Air M2 | Desktop PC | Google Colab (CPU) |
|---|---|---|---|
| SVM | 64.15 | 95.64 | 281.78 |
| Logistic Regression | 15.91 | 17.33 | 32.87 |
| Random Forest | 12.43 | 14.35 | 25.02 |
| Naive Bayes | 0.08 | 0.74 | 0.07 |

| XLNet extraction time (s) | MacBook Air M2 | Google Colab (CPU) |
|---|---|---|
| Train set | 292.38 | 1,472.92 |
| Test set | 99.80 | 472.38 |

Colab's free CPU runtime (2 cores @ 2.20 GHz) was the slowest without a GPU/TPU attached.

### Comparison with prior work

| Study | Method | Accuracy |
|---|---|---|
| Thota et al. (2018) | Word2Vec + DNN | 75.67% |
| Gupta et al. (2018) | SVM / Random Forest | 85.95% / 86.25% |
| **This project** | **FastText + XLNet + Random Forest** | **87.34%** |
| Thota et al. (2018) | TF-IDF + DNN | 94.31% |
| Moorpani et al. (2022) | Multinomial Naive Bayes | 95% |
| Prachi et al. (2022) | LSTM / BERT | 95% / 98% |

---

## Key findings

- **Random Forest gave the best balance** of accuracy and run time on every machine.
- Combining FastText with XLNet clearly beat a prior Word2Vec-only approach (87.3% vs 75.7%), supporting the idea that contextual embeddings enrich static ones.
- It still falls well short of deep learning approaches (LSTM, BERT, TF-IDF + DNN) at 94–98%.
- The cost bottleneck is XLNet feature extraction, not the classifiers.

---

## Limitations and future work

- **Tune the feature extractors**: add task-specific layers on top of XLNet, try other pre-trained FastText vectors, or train FastText on in-domain text.
- **Training techniques**: learning-rate schedules, gradient clipping, early stopping and epoch tuning.
- **Neural network classifiers**: dense, convolutional or recurrent layers on top of the combined features may make better use of them, with resource-saving strategies to keep them affordable.
- **GPU acceleration** for XLNet extraction would cut the largest time cost.

---


## Tech stack

Python · pandas · scikit-learn · PyTorch · Hugging Face Transformers (XLNet) · FastText

---

## References

- Dizikes, P. (2018). Study: On Twitter, false news travels faster than true stories. *MIT News*.
- Granik, M. & Mesyura, V. (2017). Fake news detection using naive Bayes classifier.
- Gupta, H. et al. (2018). A framework for real-time spam detection in Twitter.
- Moorpani et al. (2022). Fake news detection using Multinomial Naive Bayes.
- Prachi et al. (2022). Fake news detection using machine learning and deep learning.
- Srinivasan, A.V. (2019). *XLNet explained in simple terms*. Towards Data Science.
- Thota, A. et al. (2018). Fake news detection: A deep learning approach. *SMU Data Science Review*.
- Wang, C.-C. (2020). Fake news and related concepts: Definitions and recent research development.
- Pretrained word vector can be found here: https://fasttext.cc/docs/en/pretrained-vectors.html
