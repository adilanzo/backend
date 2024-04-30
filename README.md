<h3>Overview</h3>
This project implements a fashion product classifier using a pre-trained ResNet-18 model fine-tuned on a dataset of fashion product images. The classifier predicts the class label of an input image, which can be any type of clothing item among ten classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. The application provides a simple web interface where users can upload an image of a clothing item, and the classifier will return the predicted class label.

<h3>Instructions to Run the Application Locally:</h3>

1. Download the LAB_5.zip:  First, download the zip folder and unzip it in your desired local machine destination. 

2. Install Dependencies: Make sure you have Python installed on your machine. Navigate to the project directory and install the required Python packages using pip:
pip3 install -r requirements.txt

3. Run the Flask Application: Run the Flask application by executing the following command in your terminal:
python app.py

4. Access the Web Interface: Once the application is running, open a web browser and navigate to http://127.0.0.1:5000/ to access the web interface of the fashion product classifier.

5. Upload an Image: Click on the "Choose Image" button and select an image file of a clothing item from your local machine.

6. View Prediction: After uploading an image, the application will display the name of the uploaded file in italics and provide the predicted class label of the clothing item.

<h3>Additional Information/Observations:</h3>

• The project includes a static folder to store static assets such as images. A logo was created using Canvas and was added to enhance the visual appeal of the web interface. This logo was placed in the static folder and referenced in the HTML code using the appropriate file path.

• The design of the web application was customized to make it more appealing and user-friendly. This involved creating a visually pleasing layout, enhancing the appearance of buttons, and adding spacing between elements for better visual separation. These improvements were made to enhance the overall user experience and create a more engaging interface for interacting with the fashion product classifier.