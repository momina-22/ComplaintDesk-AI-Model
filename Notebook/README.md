---


# Google Colab Notebooks Guide

Run training and evaluation directly in the cloud using Google Colab with free GPU runtimes.

---

## Available Notebooks

| Notebook | Description | Open in Colab |
| :--- | :--- | :--- |
| **`ComplaintDesk_training_colab.ipynb`** | Fine-tunes `distilbert-base-uncased` and exports model artifacts[cite: 2]. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/ComplaintDesk/blob/main/Notebooks/ComplaintDesk_training_colab.ipynb) |
| **`ComplaintDesk_testing_colab.ipynb`** | Loads saved weights to test sample complaints and display class probabilities[cite: 3]. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/ComplaintDesk/blob/main/Notebooks/ComplaintDesk_testing_colab.ipynb) |

---

## Instructions

1. Click an **Open in Colab** badge above.
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Upload your dataset or model folder when running the corresponding notebook[cite: 2, 3].
4. Run all cells sequentially (**Runtime** > **Run all** or `Ctrl + F9`).
