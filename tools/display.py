'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

class ImageDebugger:
    def __init__(self, ccnet, dataset, data_config, device, selected_indices=None):
        self.device = device
        self.ccnet = ccnet
        self.label_size = data_config.label_size
        self.selected_indices = selected_indices if selected_indices is not None else range(len(dataset))
        self.n_canvas_col = len(selected_indices)  
        self.n_canvas_row = min(self.n_canvas_col, 4)
        stack_images, stack_labels = None, None
        n_img_w, n_img_h = data_config.obs_shape[1:]
        m_img_canvas = np.ones((n_img_h*(self.n_canvas_row+1), n_img_w*(self.n_canvas_col+1), 3))
    
        for i, idx in enumerate(selected_indices):
            images = dataset[idx][0].unsqueeze(0)
            labels = dataset[idx][1].unsqueeze(0).type(torch.float)

            if stack_images==None:
                stack_images, stack_labels = images, labels
            else:
                stack_images = torch.cat([stack_images, images], dim = 0)
                stack_labels = torch.cat([stack_labels, labels], dim = 0)

            img = np.transpose(vutils.make_grid(images[:self.n_canvas_col], padding = 0, normalize=True).numpy(), (1,2,0))
            m_img_canvas[ :n_img_h*1, n_img_w*(i+1):n_img_w*(i+2)] = img.copy()
        stack_images = stack_images.to(self.device).float()
        stack_labels = stack_labels.to(self.device).float()
        
        self.canvas_image, self.debug_images, self.debug_labels = m_img_canvas, stack_images, stack_labels 
        self.n_img_w, self.n_img_h = n_img_w, n_img_h
        
        
    def update_images(self):
        n_canvas_col, n_canvas_row = self.n_canvas_col, self.n_canvas_row
        m_img_canvas = self.canvas_image
        
        with torch.no_grad():
            explains = self.ccnet.explain(self.debug_images)
            inferred_labels = self.ccnet.reason(self.debug_images, explains)
        
        for i in range(n_canvas_row):
            selected_labels = self.debug_labels[i:i + 1,:].clone().detach().expand_as(inferred_labels)
            
            with torch.no_grad():
                generated_images = self.ccnet.produce(selected_labels, explains).cpu()
            m_img_canvas[self.n_img_h*(i+1):self.n_img_h*(i+2), self.n_img_w*1:] = \
                np.transpose(vutils.make_grid(generated_images[:n_canvas_col], padding = 0, normalize=True).numpy(), (1,2,0))

    def display_image(self, figsize=(13, 13)):
        plt.figure(figsize=figsize)
        display.clear_output(wait=True)
        plt.imshow(self.canvas_image)
        plt.axis("off")
        labels = ["Female, No-smile", "Male, No-smile", "Female, Smile", "Male, Smile"]
        for i, label in enumerate(labels):
            plt.text(60, 128 * (i + 2) - 128 // 2, label, fontsize=12, va='center', ha='center')
        plt.show()
        
        return self.canvas_image