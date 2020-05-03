from torchvision import datasets, models, transforms
from PIL import Image
from torch.autograd import Variable

formatter = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def format_image(path):
    image = Image.open(path)
    image = formatter(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image
