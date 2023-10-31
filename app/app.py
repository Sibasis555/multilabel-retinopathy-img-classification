from flask import Flask, request, render_template, send_from_directory
import mysql.connector
from PIL import Image
import numpy as np
import time
import os
import torch.nn as nn
import torch
# import torchvision 
import pickle
import cv2 
from torchvision import transforms, models

app = Flask(__name__)

connection = mysql.connector.connect(user='root', password='', host='localhost', port="3306", database='retinopathy_multilabel')
print("DB connected")

# Set the folder to store uploaded images
current_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_dir, 'artifacts')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class MultilabelClassifier(nn.Module):
    def __init__(self, n_retinopathy, n_edima):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.retinopathy = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=n_retinopathy)
        )
        self.edima = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=n_edima)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'retinopathy': self.retinopathy(x),
            'edima': self.edima(x)
            # 'epoch': self.epoch(x)
        }

def preprocess_image(image_path):
    transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    image_data=cv2.imread(image_path)
    image_data=cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)
    image_data=transform(image_data)
    image_data=image_data.unsqueeze(0)
    # print(image_data.shape)
    return image_data

retinopathy_grade = {0: 'None', 1: 'Mild DR', 2: 'Moderate DR', 3: 'Severe DR', 4: 'PDR'}
diabetic_macular_edema  = {0: 'No Referable DME', 1: 'Referable DME'}
trained_model =  pickle.load(open('messidor_retinopathy_model.pkl', 'rb'))
trained_model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    edima_pred = None
    retinopathy_pred = None
    img_path = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            # input_shape = trained_model.input_shape[1:3]  # Target input image size
            time_path = time.time()
            save_img= Image.open(image)

            img_path = os.path.join(app.config['UPLOAD_FOLDER'],f"{time_path}.jpg")
            save_img.save(img_path)

            # cursor = mysql.connection.cursor()
            cursor = connection.cursor()
            img_file=f"{time_path}.jpg"

            cursor.execute("INSERT INTO retinopathy_grades (image_file) VALUES (%s)",[img_file])

            # Preprocess the input image
            processed_img = preprocess_image(img_path)
            # Perform prediction using your prediction function
            preds = trained_model(processed_img)
            
            pred_retinopathy=int(torch.argmax(preds['retinopathy'][0]))
            pred_edima=int(torch.argmax(preds['edima'][0]))
            retinopathy_pred = retinopathy_grade[pred_retinopathy]
            edima_pred = diabetic_macular_edema[pred_edima]
            print(f"Retinopathy grade is: {retinopathy_grade[pred_retinopathy]}, Risk of makular edima is: {diabetic_macular_edema[pred_edima]}")
            
            # predicted_class_index = np.argmax(predictions[0])
            # prediction = class_labels[predicted_class_index]

            cursor.execute("UPDATE retinopathy_grades SET retinopathy_pred = %s, edima_pred = %s WHERE image_file = %s",[retinopathy_pred, edima_pred, img_file])
            connection.commit()
            cursor.close()
            return render_template("index.html", uploaded= True, retinopathy_pred = retinopathy_pred, edima_pred = edima_pred, img_path = img_path)
    # fetch all data stored in db
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM retinopathy_grades")
    birds= cursor.fetchall()
    cursor.close()
    print(birds)
    return render_template('index.html', uploaded=False)

@app.route('/<img_path>')
def send_image(img_path):
    return send_from_directory("artifacts", img_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)