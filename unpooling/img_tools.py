"""A binary to train using a single GPU.
"""

import numpy as np
import header
from PIL import Image

def process_image(img):
    """
    Rescale pixel for display
    """
    threshold=1e-6
    image_min = np.amin(img)
    image_max = np.amax(img)
    if(image_min<0):
        max_val = max(abs(image_min), abs(image_max))
        scale = 0.0 if max_val < threshold else 127.0/max_val
        offset = 128.0
    else:
        scale = 0.0 if image_max < threshold else 255.0/image_max
        offset=0.0
    img = img * scale + offset
    img = img.astype(np.uint8)
    return img

def np_to_image(save_dir, images_np, oracles_np, outputs_np, names_np):
    """

    Args:
        save_dir: Directory to save outputs (full path)
        images_np: input images
        oracle_np: oracle
        cnn_id_np: instance id
    """
    count=0
    for i in range(header.BATCH_SIZE):
        input_img_name = save_dir + '/img/_input/%d.png' %names_np[i]
        input_img = process_image(images_np[i,:,:,:])
        #input_img = np.transpose(input_img, (1, 2, 0))
        #print("input_img.shape: ", input_img.shape)
        #cv2.imwrite(oracle_img_name, oracle_img)
        result = Image.fromarray(np.squeeze(input_img))
        result.save(input_img_name)

        oracle_img_name = save_dir + '/img/_oracle/%d.png' %names_np[i]
        oracle_img = process_image(oracles_np[i,:,:,:])
        result = Image.fromarray(np.squeeze(oracle_img))
        result.save(oracle_img_name)
        #cv2.imwrite(oracle_img_name, oracle_img)

        output_img_name = save_dir + './img/_output/%d.png' %names_np[i]
        output_img = process_image(outputs_np[i,:,:,:])
        #cv2.imwrite(output_img_name, output_img)
        result = Image.fromarray(np.squeeze(output_img))
        result.save(output_img_name)

        count+=1


