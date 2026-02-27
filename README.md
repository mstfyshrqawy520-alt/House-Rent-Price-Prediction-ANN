<head>
  <meta charset="utf-8">
  <title>🏠 House Rent Price Prediction — ANN</title>
</head>
<body>
  <div align="center">
    <h1>🏠 House Rent Price Prediction</h1>
    <h3>Artificial Neural Network (ANN) — End-to-End Regression System</h3>
    <p><i>Predicting monthly house rent using tabular features and a production-ready ANN pipeline.</i></p>
    <hr style="width:60%"/>
  </div>

  <h2>🚀 نبذة Project Overview</h2>
  <p dir="rtl">
    هذا المشروع يبني نظام تنبؤي لسعر إيجار الشقق/المساكن الشهري اعتماداً على بيانات ممثلة بخصائص سكنية وديموغرافية.  
    الهدف هو تقديم Pipeline هندسي متكامل: من تنظيف وتجهيز البيانات، مروراً ببناء نموذج شبكات عصبية اصطناعية (ANN)، ثم تقييم النموذج وحفظه للاستخدام في الـinference أو نشره كخدمة ويب.
  </p>

  <h2>🎯 Objectives</h2>
  <ul>
    <li>Build a robust ANN regression model that predicts monthly rent.</li>
    <li>Provide reproducible data preprocessing and feature-engineering steps.</li>
    <li>Save trained model and scaler for production inference.</li>
    <li>Document evaluation metrics and provide suggestions for deployment.</li>
  </ul>

  <h2>🧠 Model & Method</h2>
  <p>
    <b>Model:</b> Feed-forward Artificial Neural Network (Dense layers) implemented with TensorFlow / Keras.  
    <b>Input:</b> Numerical & categorical features after encoding & scaling.  
    <b>Output:</b> Continuous value — predicted monthly rent.
  </p>

  <h2>📈 Evaluation Metrics</h2>
  <ul>
    <li>Mean Absolute Error (MAE)</li>
    <li>Root Mean Squared Error (RMSE)</li>
    <li>R² (Coefficient of Determination)</li>
  </ul>





 <h2>⚙️ How to run (Local)</h2>
  <ol>
    <li>Clone the repo:
      <pre>git clone &lt;repo-link&gt;
cd house-rent-ann</pre>
    </li>
    <li>Create virtual environment & install requirements:
      <pre>python -m venv venv
# on mac/linux
source venv/bin/activate
# on windows
venv\Scripts\activate

pip install -r requirements.txt</pre>
    </li>
    <li>Open & run the notebook:
      <pre>jupyter notebook
# then open House_rent_by_ANN.ipynb</pre>
    </li>
    <li>Or run training script (if provided):
      <pre>python src/train.py --data ../House_Rent_Dataset.csv --epochs 100 --batch_size 32</pre>
    </li>
    <li>Do inference:
      <pre>python src/inference.py --model artifacts/rent_ann_model.h5 --input '{"area":85,"rooms":2,...}'</pre>
    </li>
  </ol>

  <h2>🛠 Tech Stack</h2>
  <ul>
    <li>Python</li>
    <li>Pandas / NumPy</li>
    <li>TensorFlow / Keras (or optionally PyTorch)</li>
    <li>Scikit-learn (preprocessing, metrics)</li>
    <li>Matplotlib / Seaborn</li>
  </ul>

  <h2>💡 Engineering Notes & Best Practices</h2>
  <ul>
    <li>Split data into train/validation/test with fixed random seed for reproducibility.</li>
    <li>Scale numerical features (StandardScaler or MinMax).</li>
    <li>Encode categorical features (One-Hot / Ordinal / Target Encoding as appropriate).</li>
    <li>Use callbacks: EarlyStopping, ModelCheckpoint to avoid overfitting.</li>
    <li>Log training metrics and save training history (for plotting).</li>
    <li>Provide a small example JSON for inference in `examples/`.</li>
  </ul>

  <h2>🔮 Future Improvements</h2>
  <ul>
    <li>Hyperparameter tuning (Optuna / Keras Tuner)</li>
    <li>Model ensembling (ANN + Gradient Boosting)</li>
    <li>Deploy model as a REST API (FastAPI / Flask) with Docker</li>
    <li>Build a lightweight frontend for demo (Streamlit / Gradio)</li>
  </ul>

  <hr/>
  <div align="center">
    <h3>👨‍💻 Developed by Mostafa Sharqawy</h3>
    <p>AI Engineer | Deep Learning | Applied ML</p>
  </div>
</body>

