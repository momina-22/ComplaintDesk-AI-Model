# ComplaintDesk - Priority Classification (DistilBERT)

A text classification pipeline fine-tuned on **`distilbert-base-uncased`** to classify incoming complaints into three priority categories:

* **High**
* **Medium**
* **Low**

## Project Structure

```text
COMPLAINTDESK/
├── Final Dataset/
│   └── # Training dataset CSV files
├── Notebooks/
│   ├── README.md
│   ├── ComplaintDesk_training_colab.ipynb
│   └── ComplaintDesk_testing_colab.ipynb
├── .gitignore
├── complaintdesk_training.py
├── complaintdesk_testing.py
├── requirements.txt
└── README.md
```

## Model

The project uses the pretrained **DistilBERT** model:

```text
distilbert-base-uncased
```

The model is fine-tuned for sequence classification to predict the priority of incoming complaints.

### Classification Labels

| Priority   | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| **High**   | Complaints requiring urgent attention or immediate action   |
| **Medium** | Complaints requiring attention but not immediately critical |
| **Low**    | Complaints that can be handled through normal processing    |

## Model Weights

Model weight files such as `.safetensors` and `.bin` are **not included in this repository** due to GitHub file-size limitations.

The trained model can be obtained by either:

1. Running the training script locally.
2. Downloading the trained model from the project's linked **Hugging Face model repository**.

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<USERNAME>/ComplaintDesk.git
cd ComplaintDesk
```

Replace `<USERNAME>` with the GitHub username or organization that owns the repository.

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Locally

### Step 1: Model Training

Run the training script:

```bash
python complaintdesk_training.py
```

The script:

* Loads the dataset from `Final Dataset/`
* Preprocesses the complaint data
* Fine-tunes the DistilBERT model
* Exports the fine-tuned model weights
* Generates `label_map.json`

The trained model is saved to the configured output model directory.

### Step 2: Testing & Evaluation

Run the testing script:

```bash
python complaintdesk_testing.py
```

The script performs inference across the evaluation suite and displays the predicted priority along with confidence scores for each class.

Example:

```text
Complaint: [sample complaint]

Predicted Priority: High

Confidence Scores:
High:   0.94
Medium: 0.04
Low:    0.02
```

## Google Colab

The project includes Google Colab notebooks for training and testing.

### Training

Open:

```text
Notebooks/ComplaintDesk_training_colab.ipynb
```

### Testing

Open:

```text
Notebooks/ComplaintDesk_testing_colab.ipynb
```

For detailed instructions on running the notebooks in Google Colab, refer to:

```text
Notebooks/README.md
```

## Requirements

All required Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

## Usage Summary

### Train the Model

```bash
python complaintdesk_training.py
```

### Test the Model

```bash
python complaintdesk_testing.py
```

### Google Colab

* `Notebooks/ComplaintDesk_training_colab.ipynb`
* `Notebooks/ComplaintDesk_testing_colab.ipynb`

## Notes

* Training datasets are stored in `Final Dataset/`.
* Local training and testing are supported through Python scripts.
* Google Colab notebooks are provided for cloud-based execution.
* Large model weight files are excluded from GitHub.
* The trained model can be recreated locally using `complaintdesk_training.py`.
* Alternatively, the trained model can be downloaded from the linked Hugging Face repository.

---
## Second Method  
## `Notebooks/README.md` (Inside Notebooks Folder)

---
## Instructions to run notebooks in Colab

1. Click an **Open in Colab** badge above.
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Upload your dataset or model folder when running the corresponding notebook[cite: 2, 3].
4. Run all cells sequentially (**Runtime** > **Run all** or `Ctrl + F9`).

