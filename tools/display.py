import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image as PILImage
from IPython.display import display, Image, clear_output
import io
from PIL import Image as PILImage, ImageDraw, ImageFont

class ImageDebugger:
    def __init__(self, model, dataset, data_config, device):
        self.device = device
        self.model = model
        selected_indices = data_config.show_image_indices
        self.label_size = data_config.label_size
        self.n_canvas_col = len(selected_indices)  
        self.n_canvas_row = min(self.n_canvas_col, 4)
        stack_images, stack_labels = None, None
        n_img_w, n_img_h = data_config.obs_shape[1:]
        m_img_canvas = np.ones((n_img_h*(self.n_canvas_row+1), n_img_w*(self.n_canvas_col+1), 3), dtype=np.uint8) * 255
    
        for i, idx in enumerate(selected_indices):
            images = dataset[idx][0].unsqueeze(0)
            labels = dataset[idx][1].unsqueeze(0).type(torch.float)

            if stack_images is None:
                stack_images, stack_labels = images, labels
            else:
                stack_images = torch.cat([stack_images, images], dim=0)
                stack_labels = torch.cat([stack_labels, labels], dim=0)

            img = np.transpose(vutils.make_grid(images[:self.n_canvas_col], padding=0, normalize=True).numpy(), (1,2,0))
            m_img_canvas[:n_img_h*1, n_img_w*(i+1):n_img_w*(i+2)] = (img * 255).astype(np.uint8)
        stack_images = stack_images.to(self.device).float()
        stack_labels = stack_labels.to(self.device).float()
        
        self.canvas_image = m_img_canvas
        self.debug_images = stack_images
        self.debug_labels = stack_labels 
        self.n_img_w, self.n_img_h = n_img_w, n_img_h
        
    def update_images(self):
        n_canvas_col, n_canvas_row = self.n_canvas_col, self.n_canvas_row
        m_img_canvas = self.canvas_image
        
        with torch.no_grad():
            explains = self.model.explain(self.debug_images)
            inferred_labels = self.model.reason(self.debug_images, explains)
        
        for i in range(n_canvas_row):
            selected_labels = self.debug_labels[i:i + 1, :].clone().detach().expand_as(inferred_labels)
            
            with torch.no_grad():
                generated_images = self.model.produce(selected_labels, explains).cpu()
            img_section = np.transpose(vutils.make_grid(generated_images[:n_canvas_col], padding=0, normalize=True).numpy(), (1,2,0))
            m_img_canvas[self.n_img_h*(i+1):self.n_img_h*(i+2), self.n_img_w*1:] = (img_section * 255).astype(np.uint8)

    def display_image(self):
        """Display the image using IPython's display module with text annotations."""
        # Clear the previous output, including images, text, etc.
        clear_output(wait=True)

        # Convert numpy array to PIL Image
        img = PILImage.fromarray(self.canvas_image)
        
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