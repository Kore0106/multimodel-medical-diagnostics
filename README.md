# Multimodal Medical Diagnosis Board

<p align="left">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2500&pause=800&color=41adf2&vCenter=true&width=550&lines=Initializing+Late-Fusion+Neural+Network...;Syncing+ResNet50+Vision+Layers...;Vectorizing+SentenceTransformer+Tokens...;Deployment+Status:+ONLINE" alt="Pipeline Status Animation" />
</p>

<p align="left">
  <a href="https://isha-1410-multimodal-medical-diagnostics.hf.space" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Live_Production_App-🍏_Running-brightgreen?style=for-the-badge&logo=huggingface&logoColor=white" alt="Live Deployment Status" />
  </a>
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

An intelligent healthcare analytics engine and interactive decision-support core designed to evaluate complex, multi-tier medical data patterns. This project integrates a deep learning multimodal architecture with a secure, authenticated dashboard to deliver localized, high-efficiency clinical predictions.

---

## 🌌 System Architecture Overview

The core engine evaluates heterogeneous cross-modality clinical markers simultaneously through an integrated late-fusion neural network:

1. **Vision Stream:** Processed via a custom `ResNet50` pipeline to isolate spatial diagnostic patterns from imaging data.
2. **Textual Stream:** Synthesized using a pre-trained `SentenceTransformer` to extract semantic embeddings from clinical notes and narratives.
3. **Biomarker Stream:** Evaluated via a specialized dense tabular vector network to parse numerical laboratory results.
4. **Fusion Block:** Features robust tensor alignment and padding protection layers to safely blend feature vectors prior to classification, preventing mathematical runtime faults.

---

## ✨ Core Features

* **True Cross-Modality Fusion:** Jointly processes medical images, patient narratives, and structural laboratory reports.
* **Clinical Registry Gate:** Built-in secure authentication framework simulating multi-tier user role restrictions (e.g., `doctor`, `lead_nurse`).
* **Scannable Analytics UI:** Dense data visualization dashboard crafted via Gradio, mapping complex probability arrays into human-interpretable diagnostic charts.
* **Zero Leakage Validation Loop:** Strict separate pipeline processing to isolate evaluation arrays and protect validation data splits against engineering critique.

---

## 🧠 Cross-Modality Data Pipeline Flow

The architecture operates via a structured late-fusion pipeline that synchronizes multi-tier inputs in real time[cite: 1]:

```text
[ Patient Medical Image ]  ───────> Custom ResNet50 Stream ────────┐
                                                                   v
[ Textual Clinical Notes ] ───────> SentenceTransformer Tokens ───> [ Late-Fusion Block ] ───> [ Dense Analytics Dashboard ]
                                                                   ^
[ Lab Tabular Biomarkers ] ───────> Tabular Vector Processing ─────┘
