# Pediatric Chest X-ray Pneumonia Detection

This repository contains a PyTorch-based deep learning project to detect pneumonia from pediatric chest X-ray images. It uses a transfer-learning approach with a pre-trained **DenseNet161** model.

## 📊 Dataset
The full image dataset is securely hosted on Google Drive. 
- **[View the Dataset Here](https://drive.google.com/drive/folders/1H1-U0ocFAFhQJ_sdXiizDCIOK_lziM-k?usp=drive_link)**
- **Note for Developers**: You do **not** need to manually download this dataset! When you run the training script (`src/train.py`), the code will automatically download the entire dataset from Google Drive directly into your environment using `gdown`. This keeps the GitHub repository lightweight and professional.

## Project Structure
```text
pneumonia-detection/
│
├── data/
│   └── sample/          # Sample images for testing the frontend
│
├── src/
│   ├── train.py         # Model training script
│   ├── test.py          # Model evaluation script
│   └── utils.py         # Helper functions (transforms, model setup)
│
├── models/              # Saved model weights
│
├── images/              # Evaluation plots and confusion matrices
│
├── app.py               # Streamlit web frontend for interactive inference
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## How to Run the Training on Google Colab
Since training a deep learning model requires a GPU, the easiest way to train the model is on Google Colab:
1. Upload the entire `pneumonia-detection` folder to Google Drive.
2. Open a new Google Colab Notebook and mount your Drive.
3. Change directory into the project folder:
   ```python
   %cd /content/drive/MyDrive/pneumonia-detection
   ```
4. Install the requirements:
   ```python
   !pip install -r requirements.txt
   ```
5. Run the training script (it will automatically download the dataset from the provided Google Drive link):
   ```python
   !python src/train.py
   ```
6. Evaluate the model:
   ```python
   !python src/test.py
   ```

## How to Run the Web Frontend (Locally or Cloud)
Once you have trained the model and `best_model.pt` is saved in the `models/` directory, you can run the interactive Streamlit frontend!

If you have Python installed locally:
1. Open a terminal in the `pneumonia-detection` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Upload an X-ray image in your browser to see the prediction!

*Note: If you do not have Python installed, you can upload this repository to GitHub and connect it to [Streamlit Community Cloud](https://streamlit.io/cloud) to host the app for free online.*
