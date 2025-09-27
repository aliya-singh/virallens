# Laptop Price Prediction (Task123)

This project demonstrates a complete ML pipeline including:

- **Data Handling**
- **Model Training & Evaluation**
- **Deployment Simulation with FastAPI**

---

## 📂 Project Structure

```bash
task123/
│── requirements.txt          # Required dependencies
│── task1and2.ipynb           # Data Handling & Model Training/Evaluation
│── train_save_model.py       # Script to train and save RandomForest model
│── app.py                    # FastAPI server script
│── rf_model.pkl              # Pre-trained RandomForest model (generated after training)
│── scaler.pkl                # StandardScaler object (generated after training)
│── request.jpeg              # Sample API request screenshot
│── response.jpeg             # Sample API response screenshot
│── response.py               # Python file to get response
│── nof.py                    # Python file to count number of features
│── laptopData.csv            # .csv file 

⚙️ Setup Instructions

1️⃣ Install Dependencies
    pip install -r requirements.txt

2️⃣ Run Data Handling & Model Training
    jupyter notebook task1and2.ipynb
This notebook contains data preprocessing, training, evaluation, and observations.

3️⃣ Train & Save Model (Optional)
If rf_model.pkl and scaler.pkl are not present, run:
    python train_save_model.py
Generated files:
    rf_model.pkl → Pre-trained RandomForest model
    scaler.pkl → Fitted StandardScaler

4️⃣ Run FastAPI Server
    uvicorn app:app --reload


🚀 Testing the API
✅ Using Python File response.py
    python response.py


Sample response json:
    status Code: 200
    Response Text: {"predicted_price":72023.0}
    {'predicted_price': 72023.0}


📸 Sample request/response Screenshots
    task123/request.jpeg
    task123/response.jpeg

🛠 Tech Stack
    Python
    Pandas, NumPy
    Scikit-learn (RandomForest Regressor)
    FastAPI
    Uvicorn