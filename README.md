**ComplaintDesk - Priority Classification (DistilBERT)**
A text classification pipeline fine-tuned on distilbert-base-uncased to classify incoming complaints into priority categories: High, Medium, and Low.  Project StructurePlaintextCOMPLAINTDESK/
├── Final Dataset/                      # Contains training dataset CSV files
├── Notebooks/
│   ├── README.md                      # Guide for running in Google Colab
│   ├── ComplaintDesk_training_colab.ipynb
│   └── ComplaintDesk_testing_colab.ipynb
├── .gitignore
├── complaintdesk_training.py          # Model training script (VS Code / CLI)
├── complaintdesk_testing.py           # Model testing & inference script
├── requirements.txt                   # Project dependencies
└── README.md                          # Main project documentation
**Note:** Model weights (.safetensors / .bin) are excluded from this repository due to GitHub file size limits. You can generate them locally by running the training script or download them from the linked Hugging Face model repository.Setup & InstallationClone the repository:Bashgit clone https://github.com/<your-username>/ComplaintDesk.git
cd ComplaintDesk
**Create and activate a virtual environment:**
On Windows:
Bashpython -m venv venv
venv\Scripts\activate
On macOS / Linux:
Bashpython3 -m venv venv
source venv/bin/activate
**Install required dependencies:**
Bashpip install -r requirements.txt
**Running Locally**
**Step 1: ****Model Training**
Run complaintdesk_training.py to train the DistilBERT model on your dataset:  
Bashpython complaintdesk_training.py
Inputs: Dataset CSV located in Final Dataset/.  
Outputs: Fine-tuned model weights and label_map.json exported to the output model folder.  
**Step 2: ****Testing & Evaluation**
Run complaintdesk_testing.py to run inference across the evaluation suite and print class confidence scores:  
Bashpython complaintdesk_testing.py
