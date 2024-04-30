from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io


app = Flask(__name__)


# Load the fine-tuned model
model = models.resnet18()
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('fashion_classifier.pth', map_location=torch.device('cpu')))
model.eval()


# Define the class labels
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Define the image transformations
transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# HTML code
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
   <meta charset="UTF-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <title>Fashion Product Classifier</title>
   <style>
       body {
           font-family: Arial, sans-serif;
           text-align: center;
           background-color: #f5f5f5;
           margin: 0;
           padding: 0;
       }
       h1 {
           margin-top: 30px;
       }
       .logo {
           margin-top: 20px;
           margin-bottom: 20px;
       }
       form {
           margin-top: 20px;
       }
       #uploadForm {
           display: flex;
           flex-direction: column;
           align-items: center;
       }
       #uploadForm input[type="file"] {
           display: none;
       }
       .custom-file-upload {
           display: inline-block;
           padding: 10px 20px;
           margin-bottom: 10px; /* Added spacing between buttons */
           cursor: pointer;
           background-color: #007bff;
           color: #fff;
           border: none;
           border-radius: 5px;
           transition: background-color 0.3s ease;
       }
       .custom-file-upload:hover {
           background-color: #0056b3;
       }
       #resultContainer {
           margin-top: 20px;
       }
   </style>
</head>
<body>
   <div class="logo">
       <img src="/static/ISOM900 LOGO.png" alt="Clothing Logo" width="400">
   </div>
   <h1>Fashion Product Classifier</h1>
   <form id="uploadForm" enctype="multipart/form-data">
       <label for="fileUpload" class="custom-file-upload">Choose Image</label>
       <input type="file" id="fileUpload" name="image" accept="image/*" required>
       <button type="submit" class="custom-file-upload">Classify</button>
   </form>
   <div id="resultContainer"></div>
   <script>
       document.getElementById('uploadForm').addEventListener('submit', function(e) {
           e.preventDefault();
           var formData = new FormData();
           var fileInput = document.querySelector('input[type="file"]');
           formData.append('image', fileInput.files[0]);
           fetch('/predict', {
               method: 'POST',
               body: formData
           })
           .then(response => response.json())
           .then(data => {
               var resultContainer = document.getElementById('resultContainer');
               resultContainer.innerHTML = '<p>Predicted Class: ' + data.predicted_class + '</p>';
           })
           .catch(error => {
               console.error('Error:', error);
           });
       });

       document.getElementById('fileUpload').addEventListener('change', function() {
           var fileInput = document.getElementById('fileUpload');
           var fileName = fileInput.files[0].name;
           var resultContainer = document.getElementById('resultContainer');
           resultContainer.innerHTML = '<p><em>' + fileName + '</em> has been uploaded</p>';
       });
   </script>
</body>
</html>
'''


@app.route('/')
def index():
   return index_html


@app.route('/predict', methods=['POST'])
def predict():
   if 'image' not in request.files:
       return jsonify({'error': 'No image found'}), 400


   image = request.files['image']
   image_bytes = image.read()
   img = Image.open(io.BytesIO(image_bytes))
   img = transform(img).unsqueeze(0)


   with torch.no_grad():
       outputs = model(img)
       _, predicted = torch.max(outputs, 1)
       predicted_class = class_labels[predicted.item()]


   return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
   app.run(debug=True)
