import numpy as np
import torch
import torchvision.utils as vutils

def load_images_and_labels(dataset, indices, device, use_core):
    images = []
    labels = [] if use_core else None
    
    for idx in indices:
        img = dataset[idx][0].unsqueeze(0)
        images.append(img)
        
        if use_core:
            lbl = dataset[idx][1].unsqueeze(0).type(torch.float)
            labels.append(lbl)
    
    if use_core:
        return torch.cat(images).to(device), torch.cat(labels).to(device)
    else:
        return torch.cat(images).to(device), None

def prepare_canvas(n_img_h, n_img_w, num_images):
    return np.ones((n_img_h * (num_images + 1), n_img_w * (num_images + 1), 3), dtype=np.uint8) * 255

def place_image_on_canvas(image, canvas, n_img_h, n_img_w, i, n_img_ch, use_core):
    img = image.unsqueeze(0).cpu()
    img = vutils.make_grid(img, padding=0, normalize=True)
    img = np.transpose(img.numpy(), (1, 2, 0))
    if n_img_ch == 1:
        img = np.stack([img[:, :, 0]] * 3, axis=-1)  # Convert grayscale to RGB by repeating the channel
    image_values = (img * 255).astype(np.uint8)
    canvas[:n_img_h, n_img_w * (i + 1):n_img_w * (i + 2)] = image_values
    if not use_core:
        canvas[n_img_h * (i + 1):n_img_h * (i + 2), :n_img_w] = image_values
    return canvas

def add_celebA_labels(draw, font, n_img_w, n_img_h):
    labels = ["Female, No-smile", "Male, No-smile", "Female, Smile", "Male, Smile"]
    positions = [(n_img_w // 4, n_img_h // 2 + n_img_h * (i + 1)) for i in range(len(labels))]
    for label, position in zip(labels, positions):
        draw.text(position, label, font=font, fill=(0, 0, 0))

def add_mnist_labels(draw, font, n_img_w, n_img_h):
    labels = ["style", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    positions = [(n_img_w // 8, n_img_h // 4 + n_img_h * i) for i in range(len(labels))]
    for label, position in zip(labels, positions):
        draw.text(position, label, font=font, fill=(0, 0, 0))

def add_celebA_ukiyoe_labels(draw, font, n_img_w, n_img_h):
    labels = ["Male photo", "Female photo", "Old painting", "Old painting"]
    positions = [(n_img_w // 4, n_img_h // 2 + n_img_h * (i + 1)) for i in range(len(labels))]
    for label, position in zip(labels, positions):
        draw.text(position, label, font=font, fill=(0, 0, 0))

def add_celebA_animal_labels(draw, font, n_img_w, n_img_h):
    labels = ["Man", "Woman", "Dog", "Cat"]
    positions = [(n_img_w // 4, n_img_h // 2 + n_img_h * (i + 1)) for i in range(len(labels))]
    for label, position in zip(labels, positions):
        draw.text(position, label, font=font, fill=(0, 0, 0))

def add_default_labels(draw, font, n_img_w, n_img_h):
    labels = ["Explain from the right \n Feature from below"]
    positions = [(n_img_w // 8, n_img_h // 2 + n_img_h * i) for i in range(len(labels))]
    for label, position in zip(labels, positions):
        draw.text(position, label, font=font, fill="black")
        
def text_on_image(draw, font, n_img_w, n_img_h, dataset_name, use_core):
    if use_core:
        if dataset_name == 'celebA':
            add_celebA_labels(draw, font, n_img_w, n_img_h)
        elif dataset_name == 'mnist':
            add_mnist_labels(draw, font, n_img_w, n_img_h)
        elif dataset_name == 'celebA_ukiyoe':
            add_celebA_ukiyoe_labels(draw, font, n_img_w, n_img_h)
        elif dataset_name == 'celebA_animal':
            add_celebA_animal_labels(draw, font, n_img_w, n_img_h)
        else:
            add_default_labels(draw, font, n_img_w, n_img_h)
    else:
        add_default_labels(draw, font, n_img_w, n_img_h)
        