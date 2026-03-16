# 🌟 Vietnamese Multimodal Aspect-Category Sentiment Analysis (MACSA) with Implicit Aspects

This project focuses on tackling the "implicit aspects" challenge in Multimodal Aspect-Category Sentiment Analysis (MACSA). Implicit aspects are target objects that only appear visually in images but are not explicitly mentioned in the accompanying text, causing semantic disconnection for traditional sentiment analysis models.

We propose a novel approach by integrating a Sequence-to-Sequence (Seq2Seq) pre-training strategy into the **Fine-Grained Cross-Modal Fusion (FCMF)** framework to enhance the semantic alignment between visual and textual modalities.

## 👥 Authors
* **Son Truong Thai Bui**
* **Nguyen Dac Luong**
* **Long Hoang Ung**

*(Faculty of Information Science and Engineering, University of Information Technology, VNU-HCM)*

## 🎯 Key Contributions
* **Identifying the Challenge:** Analysis of the "implicit aspect" problem in Vietnamese MACSA and identification of the limitations of current multimodal approaches.
* **Novel Architecture:** Proposal of a pre-training task called **Implicit Aspect Opinion Generation (IAOG)** based on a Seq2Seq architecture. This forces the model to actively learn how to generate descriptive sentiment words corresponding to the hidden entities detected in the visual data.
* **State-of-the-Art (SOTA) Performance:** The FCMF model integrated with IAOG significantly improves performance over the original FCMF and outperforms current SOTA architectures (mRoBERTa, tomRoBERTa, EF-CapTrRoBERTa) on implicit aspect datasets.

## ⚙️ System Architecture

The system operates in two main phases:

1.  **Phase 1 (Pre-training):** Transformation of the FCMF framework into a Seq2Seq architecture by adding a Decoder to perform the IAOG task. This aligns the hidden aspects and sentiment words.
2.  **Phase 2 (Fine-tuning):** The Decoder is removed, and the knowledge-enriched FCMF Encoder is used directly for the main Vietnamese MACSA classification task.

## 📊 Experimental Results

Experiments were conducted on a subset containing implicit aspects extracted from the ViMACSA dataset. 

| Models | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: |
| mRoBERTa | 50.80 | 52.38 | 42.98 |
| tomRoBERTa | 67.31 | 66.85 | 63.16 |
| EF-CapTrRoBERTa | **71.93** | 65.28 | 65.75 |
| FCMF | 67.47 | 70.12 | 68.77 |
| **FCMF + IAOG (Ours)** | 70.51 | **74.12** | **72.27** |

**Insight:** The proposed method shows exceptional capability in highly challenging categories with a large number of implicit aspects, such as *Public area*.

## 📂 Datasets
The datasets have been cleaned and processed for both phases of the project:
* **IAOG Pre-training Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/snbitrngthi/iaog-filtered)
* **Implicit ViMACSA Experimental Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/snbitrngthi/implicit-vimacsa)

## 🚀 Setup & Run

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/Multimodal-Aspect-Category-Sentiment-Analysis.git](https://github.com/your-username/Multimodal-Aspect-Category-Sentiment-Analysis.git)
cd Multimodal-Aspect-Category-Sentiment-Analysis

# 2. Install required packages
pip install -r requirements.txt

# 3. Run Pre-training (IAOG)
python run_pretraining_fcmf.py

# 4. Run Main MACSA Fine-tuning
python run_multimodal_fcmf.py
