import numpy as np
import torch
import torchvision.utils as vutils
import io
from IPython.display import display, Image, clear_output
from PIL import Image as PILImage, ImageDraw, ImageFont

class ImageDebugger:
    def __init__(self, image_ccnet, data_config, device, use_core=False):
        self.device = device
        self.image_ccnet = image_ccnet
        self.use_core = use_core
        self.show_image_indices = data_config.show_image_indices
        self.label_size = data_config.label_size
        self.n_img_w, self.n_img_h = data_config.obs_shape[1:]
        self.num_images = len(self.show_image_indices)

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
            img = np.transpose(vutils.make_grid(image.unsqueeze(0).cpu(), padding=0, normalize=True).numpy(), (1, 2, 0))
            self.m_canvas[:self.n_img_h, self.n_img_w * (i + 1):self.n_img_w * (i + 2)] = (img * 255).astype(np.uint8)

    def update_images(self):
        with torch.no_grad():
            if self.use_core:
                explains = self.image_ccnet.explain(self.debug_images)
                inferred_labels = self.image_ccnet.reason(self.debug_images, explains)
            else:
                image_code = self.image_ccnet.encode(self.debug_images)
                recognized_features = image_code[:, :self.model.stoch_size].clone().detach()
                explains = image_code[:, self.image_ccnet.stoch_size:].clone().detach()

            for i in range(self.num_images):
                if self.use_core:
                    selected_features = self.debug_labels[i:i + 1, :].expand_as(inferred_labels)
                    generated_images = self.image_ccnet.produce(selected_features, explains).cpu()
                else:
                    selected_features = self.debug_labels[i:i + 1, :].expand_as(inferred_labels) if self.use_core else \
                                        recognized_features[i:i + 1, :].expand_as(recognized_features)
                    generated_code = torch.cat([selected_features, explains], dim=-1)
                    generated_images = self.image_ccnet.decode(generated_code).cpu()

                img_array = np.transpose(vutils.make_grid(generated_images, padding=0, normalize=True).numpy(), (1, 2, 0))
                self.m_canvas[self.n_img_h * (i + 1):self.n_img_h * (i + 2), self.n_img_w:] = (img_array * 255).astype(np.uint8)

    def display_image(self):
        """Display the image using IPython's display module with text annotations."""
        # Clear the previous output, including images, text, etc.
        clear_output(wait=True)

        # Convert numpy array to PIL Image
        img = PILImage.fromarray(self.m_canvas)
        
        # Prepare to draw on the image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # Load default font
        
        # Define labels and their positions
        labels = ["Female, No-smile", "Male, No-smile", "Female, Smile", "Male, Smile"]
        positions = [(30, 60 + 128 * (i + 1)) for i in range(len(labels))]  # Adjust positions as needed
        
        # Draw text on the image
        for label, position in zip(labels, positions):
            draw.text(position, label, font=font, fill=(0, 0, 0))  # White color for text

        # Save the image to a byte buffer to then display it
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            display(Image(data=output.getvalue(), format="png"))