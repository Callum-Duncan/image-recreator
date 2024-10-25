
# Image recreator

Recreate images using a collection of smaller images also know as `image to emoji`, `image Mosaics`, `image recreator`

![output_img](https://github.com/user-attachments/assets/3aef4594-9e2f-4811-8c45-6ced61318b2d)

## Installation

clone the repository

```bash
  git clone https://github.com/Callum-Duncan/image-recreator.git
  cd image-recreator
  pip install -r requirements.txt
```
Or try it online:
[https://code.cratior.me/image/](code.cratior.me/image/)

# Optimizations
`metrics from tqdm`

| Metric                         | Old - Script 2                  | New - Script 13                 |
|--------------------------------|---------------------------------|---------------------------------|
| Precalculating Part Info       | 2251 [00:06] 327.72 it/s        | 2251  [00:00] 2,273.73     it/s |
| Extracting Regions             |                                 | 54000 [00:00] 3,176,522.62 it/s |
| Calculating Average RGBs       |                                 | 54000 [00:00] 279,792.98   it/s |
| Finding Closest Matching Parts |                                 | 54000 [00:00] 194,946.05   it/s |
| Placing Closest Parts          | 54000 [00:12] 497.53 it/s       | 54000 [00:00] 1,710,631.33   it/s |

# Image Settings Configuration

## Memory and Processing Requirements

Larger image widths require significantly more memory and processing time.

### For 8 GB of RAM:
- **Suggested upper limit**: The system can handle more, but this is a good limit.
- **Designed width**: `40 * 2000`
- **Max threads**: It is not recommended to use all available threads. With 8 threads, use 5.

---

## Explanation of Settings

### `Images`:
- **Folder Path**: The folder containing all the images that will be used to recreate the input.
- **Default**: `"imgs"`

### `desired_width`:
- **Width of the Output Image**: 
    - The target width of the image, defined as the number of images wide multiplied by the image width.
    - **Example**: `32 * 300`

### `part_size`:
- **Image Resize Factor**:
    - Used to resize the images.
    - **Default**: `32`

### `max_threads`:
- **Number of Threads to Use**:
    - The number of threads for processing.
    - **Recommendation**: For 8 threads, use a maximum of 5.
    - **Default**: `5`

### `Output_image_width`:
- **Width of the Final Output Image**:
    - This setting defines the width of the recreated output image. Increasing this value will improve resolution but also require more resources.
    - A good high value is around `20,000`.
    - **Default**: `5000`

