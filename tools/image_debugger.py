import numpy as np
import torch
import torchvision.utils as vutils
import io
import base64
from IPython.display import display, clear_output, Image, HTML
from PIL import Image as PILImage, ImageDraw, ImageFont


# self.n_img_ch
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

    def _load_images_and_labels(self, dataset, indices):
        images, labels = [], []
        for idx in indices:
            img, lbl = dataset[idx]
            images.append(img.unsqueeze(0))
            labels.append(lbl.unsqueeze(0).type(torch.float))
        return torch.cat(images).to(self.device), torch.cat(labels).to(self.device)

    def initialize_(self, dataset):
        self.debug_images, self.debug_labels = self._load_images_and_labels(dataset, self.show_image_indices)
        self.m_canvas = np.ones((self.n_img_h * (self.num_images + 1), self.n_img_w * (self.num_images + 1), 3), dtype=np.uint8) * 255

        for i, image in enumerate(self.debug_images):
            img = image.unsqueeze(0).cpu()
            img = vutils.make_grid(img, padding=0, normalize=True)
            img = np.transpose(img.numpy(), (1, 2, 0))
            if self.n_img_ch == 1:
                img = np.stack([img[:, :, 0]] * 3, axis=-1)  # Convert grayscale to RGB by repeating the channel
            self.m_canvas[:self.n_img_h, self.n_img_w * (i + 1):self.n_img_w * (i + 2)] = (img * 255).astype(np.uint8)

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
            if self.dataset_name == 'celebA':
                font = ImageFont.load_default()  # Load default font

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

        # Save the image to a byte buffer to then display it using HTML
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            data_uri = base64.b64encode(output.getvalue()).decode('utf-8')
            img_tag = f'<img src="data:image/png;base64,{data_uri}" style="width: {self.max_display_size}px; height: {self.max_display_size}px;" />'  # Adjust width as necessary
            display(HTML(img_tag))
            
        return img