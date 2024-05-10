from flask import Flask, request, render_template
from PIL import Image
import torch
import pickle
import io
from torchvision import transforms

app = Flask(__name__)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



import os

# Get the current directory of the app.py file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the model file relative to the current directory
model_path = os.path.join(current_directory, 'models', 'tb_model.pkl')

# Load the model from the pickle file
with open(model_path, 'rb') as f:
    model = CPU_Unpickler(f).load()


# Define the preprocessing transformations
pre_processing = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define a function to preprocess the image and make predictions
def predict_image(image, model, transform):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(image)
        # Perform necessary operations on outputs (if any)
        # For example, for binary classification, you might apply sigmoid activation:
        predicted = torch.sigmoid(outputs)
        # Convert the output to a human-readable format (e.g., class labels)
        # For binary classification, you might round the output to 0 or 1
        predicted = torch.round(predicted)
    return predicted.item() 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file)
        predicted_class = predict_image(img, model, pre_processing)
        print("Predicted class:", predicted_class)

        return render_template('result.html', predicted_class=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5003)
