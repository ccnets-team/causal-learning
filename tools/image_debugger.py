import numpy as np
import torch
import torchvision.utils as vutils
import io
import base64
from IPython.display import display, clear_output, Image, HTML
from PIL import Image as PILImage, ImageDraw, ImageFont
import os

class ImageDebugger:
    def __init__(self, image_ccnet, data_config, device, use_core=False):
        self.device = device
        self.image_ccnet = image_ccnet
        self.use_core = use_core
        self.show_image_indices = data_config.show_image_indices
        self.label_size = data_config.label_size
        self.n_img_ch, self.n_img_h, self.n_img_w = data_config.obs_shape
        self.dataset_name = data_config.dataset_name
        self.num_images = len(self.show_image_indices)

        # Set maximum display width
        self.max_display_size = 800
        self.use_save_image = False
        self.debug_iter = 0
        if self.use_save_image:
            image_save_path = self.dataset_name
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)  # Create the directory if it does not exist
            self.image_save_path = image_save_path
            self.save_image_interval = 10
        else:
            self.save_image_interval = None
            self.image_save_path = None

    def _load_images_and_labels(self, dataset, indices):
        images = []
        labels = [] if self.use_core else None
        
        for idx in indices:
            img = dataset[idx][0].unsqueeze(0)
            images.append(img)
            
            if self.use_core:
                lbl = dataset[idx][1].unsqueeze(0).type(torch.float)
                labels.append(lbl)
        
        if self.use_core:
            return torch.cat(images).to(self.device), torch.cat(labels).to(self.device)
        else:
            return torch.cat(images).to(self.device), None

    def initialize_(self, dataset):
        self.debug_images, self.debug_labels = self._load_images_and_labels(dataset, self.show_image_indices)
        self.m_canvas = np.ones((self.n_img_h * (self.num_images + 1), self.n_img_w * (self.num_images + 1), 3), dtype=np.uint8) * 255

        for i, image in enumerate(self.debug_images):
            img = image.unsqueeze(0).cpu()
            img = vutils.make_grid(img, padding=0, normalize=True)
            img = np.transpose(img.numpy(), (1, 2, 0))
            if self.n_img_ch == 1:
                img = np.stack([img[:, :, 0]] * 3, axis=-1)  # Convert grayscale to RGB by repeating the channel
            image_values = (img * 255).astype(np.uint8)
            self.m_canvas[:self.n_img_h, self.n_img_w * (i + 1):self.n_img_w * (i + 2)] = image_values
            if not self.use_core:
                self.m_canvas[self.n_img_h * (i + 1):self.n_img_h * (i + 2), :self.n_img_w] = image_values

    def update_images(self):
        with torch.no_grad():
            if self.use_core:
                explains = self.image_ccnet.explain(self.debug_images)
                inferred_labels = self.image_ccnet.reason(self.debug_images, explains)
            else:
                image_code = self.image_ccnet.encode(self.debug_images)
                recognized_features = image_code[:, :self.image_ccnet.stoch_size].clone().detach()
                explains = image_code[:, self.image_ccnet.stoch_size:].clone().detach()

            # Assuming `self.n_img_h` and `self.n_img_w` are the correct dimensions for each image
            for i in range(self.num_images):
                if self.use_core:
                    selected_features = self.debug_labels[i:i + 1, :].expand_as(inferred_labels)
                    generated_images = self.image_ccnet.produce(selected_features, explains).cpu()
                else:
                    selected_features = recognized_features[i:i + 1, :].expand_as(recognized_features)
                    generated_code = torch.cat([selected_features, explains], dim=-1)
                    generated_images = self.image_ccnet.decode(generated_code).cpu()
                if self.n_img_ch == 1:
                    generated_images = generated_images.repeat_interleave(3, dim=1)
                img_array = vutils.make_grid(generated_images, nrow = self.num_images, padding=0, normalize=True).numpy()
                img_array = np.transpose(img_array, (1, 2, 0))
                # Correcting indices for placing images
                target_row_start = self.n_img_h * (i + 1)
                target_row_end = self.n_img_h * (i + 2)
                target_col_start = self.n_img_w * (1)
                target_col_end = self.n_img_w * (self.num_images + 1)

                # Insert into canvas
                self.m_canvas[target_row_start:target_row_end, target_col_start:target_col_end] =  (img_array * 255).astype(np.uint8)

    def display_image(self):
        """Display the image using IPython's display module with HTML for better control over image size and appearance."""
        clear_output(wait=True)

        # Convert numpy array to PIL Image
        img = PILImage.fromarray(self.m_canvas)

        # Prepare to draw on the image
        draw = ImageDraw.Draw(img)

        if self.use_core:
            font = ImageFont.load_default()  # Load default font
            if self.dataset_name == 'celebA':

                # Define labels and their positions
                labels = ["Female, No-smile", "Male, No-smile", "Female, Smile", "Male, Smile"]
                positions = [(self.n_img_w//4, self.n_img_h//2 + self.n_img_h* (i + 1)) for i in range(len(labels))]  # Adjust positions as needed

                # Draw text on the image
                for label, position in zip(labels, positions):
                    draw.text(position, label, font=font, fill=(0, 0, 0))  # Black color for text
            elif self.dataset_name == 'mnist':
                # Load a specific font with a defined size
                font = ImageFont.truetype("arial.ttf", 12)  # Adjust the font and size as needed

                # Define labels and their positions
                labels = ["style", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
                positions = [(self.n_img_w//8, self.n_img_h//4 + self.n_img_h* i) for i in range(len(labels))]  # Adjust positions as needed

                # Draw text on the image
                for label, position in zip(labels, positions):
                    draw.text(position, label, font=font, fill=(0, 0, 0))  # Black color for text
                    
            elif self.dataset_name == 'celebA_ukiyoe':

                # Define labels and their positions
                labels = ["Male photo", "Female photo", "Old painting", "Old painting"]
                positions = [(self.n_img_w//4, self.n_img_h//2 + self.n_img_h* (i + 1)) for i in range(len(labels))]  # Adjust positions as needed

                # Draw text on the image
                for label, position in zip(labels, positions):
                    draw.text(position, label, font=font, fill=(0, 0, 0))  # Black color for text      
            elif self.dataset_name == 'celebA_animal':
                font = ImageFont.truetype("arial.ttf", 16)  # Adjust the font and size as needed
                # Define labels and their positions
                labels = ["Man", "Woman", "Dog", "Cat"]
                positions = [(self.n_img_w//4, self.n_img_h//2 + self.n_img_h* (i + 1)) for i in range(len(labels))]  # Adjust positions as needed

                # Draw text on the image
                for label, position in zip(labels, positions):
                    draw.text(position, label, font=font, fill=(0, 0, 0))  # Black color for text                          
        else:
            # Assume 'labels' list contains only one label for simplicity here
            labels = ["Explain from the right \n Feature from below"]
            # Calculate positions based on the image dimensions
            positions = [(self.n_img_w // 8, self.n_img_h // 2 + self.n_img_h * i) for i in range(len(labels))]
            # Draw the text on the image
            for label, position in zip(labels, positions):
                draw.text(position, label, font=font, fill="black")

        # Save the image to a byte buffer to then display it using HTML
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            data_uri = base64.b64encode(output.getvalue()).decode('utf-8')
            img_tag = f'<img src="data:image/png;base64,{data_uri}" style="width: {self.max_display_size}px; height: {self.max_display_size}px;" />'  # Adjust width as necessary
            display(HTML(img_tag))

        if self.use_save_image and self.debug_iter % self.save_image_interval == 0:
            filename = os.path.join(self.image_save_path, f'image_{self.debug_iter}.png')  # Construct the full file path
            img.save(filename, format='PNG')  # Save as PNG

        self.debug_iter += 1

        return img