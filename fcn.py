import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# Load a pre-trained FCN model
model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
model.eval()

# Preprocess the input image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load and preprocess the image
image_path = 'image.jpg'
image = Image.open(image_path).convert("RGB")
input_image = transform(image)
input_image = input_image.unsqueeze(0)

# Run the image through the model
with torch.no_grad():
    output = model(input_image)['out']

probs = torch.softmax(output, dim=1)
_, predicted_class = torch.max(probs, dim=1)

# Visualize the input image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Input Image')
plt.axis('off')

# Visualize the predicted segmentation mask
semantic_segmentation = predicted_class.squeeze().cpu().numpy()
plt.subplot(1, 2, 2)
plt.imshow(semantic_segmentation, cmap='jet')
plt.title('Semantic Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()