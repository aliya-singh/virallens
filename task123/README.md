'''# 💻 Laptop Price Prediction (Task123)

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
⚙️ Setup Instructions
1️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
2️⃣ Run Data Handling & Model Training
bash
Copy code
jupyter notebook task1and2.ipynb
This notebook contains data preprocessing, training, evaluation, and observations.

3️⃣ Train & Save Model (Optional)
If rf_model.pkl and scaler.pkl are not present, run:

bash
Copy code
python train_save_model.py
Generated files:

rf_model.pkl → Pre-trained RandomForest model

scaler.pkl → Fitted StandardScaler

4️⃣ Run FastAPI Server
bash
Copy code
uvicorn app:app --reload
Server: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

🚀 Testing the API
✅ Option 1: Using Python
python
Copy code
import requests

url = "http://127.0.0.1:8000/predict"
data = {"features": [1, 2, 15.6, 3, 5, 8, 2, 4, 1, 2.3, 0]}  # 11 features
response = requests.post(url, json=data)
print(response.json())
Expected response:

json
Copy code
{
  "predicted_price": 56789.45
}
✅ Option 2: Using Swagger UI
Open: http://127.0.0.1:8000/docs

Select /predict POST endpoint

Enter your JSON array of features

Click Try it out

📸 Sample Screenshots
Request Example


Response Example


🛠 Tech Stack
Python

Pandas, NumPy

Scikit-learn (RandomForest Regressor)

FastAPI

Uvicorn'''