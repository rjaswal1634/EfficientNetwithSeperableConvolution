## Data Visualization


```python
pip install seaborn
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting seaborn
      Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from seaborn) (1.26.4)
    Collecting pandas>=1.2 (from seaborn)
      Downloading pandas-2.2.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (89 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.9/89.9 kB[0m [31m566.3 kB/s[0m eta [36m0:00:00[0m1m554.2 kB/s[0m eta [36m0:00:01[0m
    [?25hRequirement already satisfied: matplotlib!=3.6.1,>=3.4 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from seaborn) (3.10.1)
    Requirement already satisfied: contourpy>=1.0.1 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.56.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /usr/lib/python3/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.0)
    Requirement already satisfied: pillow>=8 in /usr/lib/python3/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (10.2.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/lib/python3/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas>=1.2->seaborn) (2024.1)
    Collecting tzdata>=2022.7 (from pandas>=1.2->seaborn)
      Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
    Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m294.9/294.9 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m[31m1.6 MB/s[0m eta [36m0:00:01[0m
    [?25hDownloading pandas-2.2.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.7 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.7/12.7 MB[0m [31m31.8 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hDownloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m347.8/347.8 kB[0m [31m34.3 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: tzdata, pandas, seaborn
    Successfully installed pandas-2.2.3 seaborn-0.13.2 tzdata-2025.2
    Note: you may need to restart the kernel to use updated packages.



```python
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set up paths and categories
data_dir = "data"
classes = ["cloudy", "desert", "green_area", "water"]
output_dir = "visuals"
os.makedirs(output_dir, exist_ok=True)

# 1. Count images in each category
image_counts = {}
for cls in classes:
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    image_counts[cls] = len(images)

# 2. Bar plot: Distribution of Images
plt.figure(figsize=(10, 6))  # Slightly wider figure

# Updated barplot call using hue parameter and legend=False
sns.barplot(
    x=list(image_counts.values()), 
    y=list(image_counts.keys()), 
    hue=list(image_counts.keys()), 
    palette="viridis",
    legend=False
)

plt.title("Distribution of Images Across Categories", fontsize=14)
plt.xlabel("Number of Images", fontsize=12)
plt.ylabel("Category", fontsize=12)

# Calculate the maximum bar length to better position text
max_count = max(image_counts.values())
padding = max_count * 0.05  # 5% padding

# Position text with better spacing
for i, count in enumerate(image_counts.values()):
    plt.text(count + padding, i, str(count), va="center", fontsize=12)

# Make sure the x-axis extends beyond the longest bar to accommodate text
plt.xlim(0, max_count * 1.15)  # Add 15% extra space for labels

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distribution_barplot.jpg"), format="jpg", dpi=300)
plt.show()


## 3. Pie chart: Image Distribution
plt.figure(figsize=(8, 8))  # Square figure for better pie chart proportions
plt.pie(
    image_counts.values(),
    labels=image_counts.keys(),
    autopct="%1.1f%%",
    colors=sns.color_palette("viridis", len(classes)),
    startangle=90  # Start at top
)
plt.title("Image Distribution by Category (Percentage)", fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distribution_piechart.jpg"), format="jpg", dpi=300)
plt.show()

# 4. Sample Images from Each Category
num_samples = 3
fig = plt.figure(figsize=(12, 8))
for i, cls in enumerate(classes):
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    for j in range(min(num_samples, len(images))):
        img_path = os.path.join(class_path, images[j])
        img = Image.open(img_path)
        plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
        plt.imshow(img)
        plt.title(cls, fontsize=10)
        plt.axis("off")

plt.suptitle("Sample Images from Each Category", fontsize=16, y=1.02)
plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout while preserving space for suptitle
plt.savefig(os.path.join(output_dir, "sample_images.jpg"), format="jpg", dpi=300)
plt.show()

```


    
![png](output_files/output_2_0.png)
    



    
![png](output_files/output_2_1.png)
    



    
![png](output_files/output_2_2.png)
    


The dataset, sourced from mahmoudreda55/satellite-image-classification on Kaggle, contains 5631 satellite images across four categories: cloudy, desert, green area, and water. The distribution reveals an imbalance, with cloudy, green area, and water each comprising 1500 images (26.6%), while desert has 1131 images (20.1%), as shown in Figure 1. This disparity may lead to better model performance on the balanced classes, with potential challenges in classifying desert images accurately. Data augmentation was applied to mitigate this imbalance during training.


```python
# Data Augmentation
```


```python
pip install albumentations
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting albumentations
      Downloading albumentations-2.0.5-py3-none-any.whl.metadata (41 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.7/41.7 kB[0m [31m754.2 kB/s[0m eta [36m0:00:00[0mMB/s[0m eta [36m0:00:01[0m
    [?25hRequirement already satisfied: numpy>=1.24.4 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from albumentations) (1.26.4)
    Requirement already satisfied: scipy>=1.10.0 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from albumentations) (1.15.2)
    Requirement already satisfied: PyYAML in /usr/lib/python3/dist-packages (from albumentations) (6.0.1)
    Collecting pydantic>=2.9.2 (from albumentations)
      Downloading pydantic-2.11.3-py3-none-any.whl.metadata (65 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.2/65.2 kB[0m [31m1.1 MB/s[0m eta [36m0:00:00[0m[31m4.4 MB/s[0m eta [36m0:00:01[0m
    [?25hCollecting albucore==0.0.23 (from albumentations)
      Downloading albucore-0.0.23-py3-none-any.whl.metadata (5.3 kB)
    Collecting opencv-python-headless>=4.9.0.80 (from albumentations)
      Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
    Collecting stringzilla>=3.10.4 (from albucore==0.0.23->albumentations)
      Downloading stringzilla-3.12.3-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_28_x86_64.whl.metadata (80 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m80.3/80.3 kB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting simsimd>=5.9.2 (from albucore==0.0.23->albumentations)
      Downloading simsimd-6.2.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (66 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m66.0/66.0 kB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations)
      Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
    Collecting pydantic-core==2.33.1 (from pydantic>=2.9.2->albumentations)
      Downloading pydantic_core-2.33.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
    Collecting typing-extensions>=4.12.2 (from pydantic>=2.9.2->albumentations)
      Downloading typing_extensions-4.13.1-py3-none-any.whl.metadata (3.0 kB)
    Collecting typing-inspection>=0.4.0 (from pydantic>=2.9.2->albumentations)
      Downloading typing_inspection-0.4.0-py3-none-any.whl.metadata (2.6 kB)
    Downloading albumentations-2.0.5-py3-none-any.whl (290 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m290.6/290.6 kB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m MB/s[0m eta [36m0:00:01[0m
    [?25hDownloading albucore-0.0.23-py3-none-any.whl (14 kB)
    Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.0/50.0 MB[0m [31m26.1 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hDownloading pydantic-2.11.3-py3-none-any.whl (443 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m443.6/443.6 kB[0m [31m27.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pydantic_core-2.33.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m34.9 MB/s[0m eta [36m0:00:00[0m31m45.2 MB/s[0m eta [36m0:00:01[0m
    [?25hDownloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
    Downloading simsimd-6.2.1-cp312-cp312-manylinux_2_28_x86_64.whl (633 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m633.1/633.1 kB[0m [31m22.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading stringzilla-3.12.3-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_28_x86_64.whl (308 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m308.2/308.2 kB[0m [31m40.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading typing_extensions-4.13.1-py3-none-any.whl (45 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.7/45.7 kB[0m [31m9.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading typing_inspection-0.4.0-py3-none-any.whl (14 kB)
    Installing collected packages: stringzilla, simsimd, typing-extensions, opencv-python-headless, annotated-types, typing-inspection, pydantic-core, albucore, pydantic, albumentations
    Successfully installed albucore-0.0.23 albumentations-2.0.5 annotated-types-0.7.0 opencv-python-headless-4.11.0.86 pydantic-2.11.3 pydantic-core-2.33.1 simsimd-6.2.1 stringzilla-3.12.3 typing-extensions-4.13.1 typing-inspection-0.4.0
    Note: you may need to restart the kernel to use updated packages.



```python
import os
import numpy as np
from PIL import Image
import albumentations as A
import random

# Path to the dataset
data_dir = os.path.join("data")
classes = ["cloudy", "desert", "green_area", "water"]

# Count images in each category
image_counts = {}
for cls in classes:
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    image_counts[cls] = len(images)

print("Original counts:", image_counts)

# Target count (match other classes)
target_count = 1500
desert_images = [f for f in os.listdir(os.path.join(data_dir, "desert")) if f.endswith(".jpg")]
num_to_generate = target_count - len(desert_images)  # 369 images needed

# Define augmentation pipeline
augmentations = A.Compose([
    A.Rotate(limit=90, p=0.5),  # Rotate up to 90 degrees
    A.HorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    A.VerticalFlip(p=0.5),  # 50% chance of vertical flip
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjust brightness/contrast
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Color jitter
    A.RandomCrop(height=200, width=200, p=0.3),  # Random crop (then resize back)
    A.Resize(height=224, width=224),  # Resize to match EfficientNet input
])

# Generate new desert images
generated_count = 0
desert_path = os.path.join(data_dir, "desert")
while generated_count < num_to_generate:
    # Randomly select an image to augment
    img_name = random.choice(desert_images)
    img_path = os.path.join(desert_path, img_name)
    
    # Load image
    img = np.array(Image.open(img_path))
    
    # Apply augmentations
    augmented = augmentations(image=img)
    aug_img = augmented["image"]
    
    # Save the augmented image with a new name
    base_name, ext = os.path.splitext(img_name)
    new_img_name = f"{base_name}_aug_{generated_count}{ext}"
    new_img_path = os.path.join(desert_path, new_img_name)
    Image.fromarray(aug_img).save(new_img_path)
    
    generated_count += 1

# Verify new counts
new_counts = {}
for cls in classes:
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    new_counts[cls] = len(images)

print("New counts after augmentation:", new_counts)
```

    Original counts: {'cloudy': 1500, 'desert': 1131, 'green_area': 1500, 'water': 1500}
    New counts after augmentation: {'cloudy': 1500, 'desert': 1500, 'green_area': 1500, 'water': 1500}


## Distrubution After Augmentation


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Recount images
image_counts = {}
for cls in classes:
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    image_counts[cls] = len(images)


plt.figure(figsize=(10, 6))  # Slightly wider figure

sns.barplot(
    x=list(image_counts.values()), 
    y=list(image_counts.keys()), 
    hue=list(image_counts.keys()), 
    palette="viridis",
    legend=False
)

plt.title("Distribution of Images Across Categories After Augmentation", fontsize=14)
plt.xlabel("Number of Images", fontsize=12)
plt.ylabel("Category", fontsize=12)

# Calculate the maximum bar length to better position text
max_count = max(image_counts.values())
padding = max_count * 0.05  # 5% padding

# Position text with better spacing
for i, count in enumerate(image_counts.values()):
    plt.text(count + padding, i, str(count), va="center", fontsize=12)

# Make sure the x-axis extends beyond the longest bar to accommodate text
plt.xlim(0, max_count * 1.15)  # Add 15% extra space for labels

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distribution_barplot_aug.jpg"), format="jpg", dpi=300)
plt.show()
```


    
![png](output_files/output_8_0.png)
    


# Sprectrum Analysis


```python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the dataset
data_dir = os.path.join("data")
classes = ["cloudy", "desert", "green_area", "water"]

# Number of samples per class (1 for simplicity)
num_samples = 1

# Set up the plot
fig, axes = plt.subplots(len(classes), 4, figsize=(12, 8))  # 4 columns: original, R, G, B
plt.suptitle("RGB Spectrum Analysis of Sample Images by Category", fontsize=16, y=1.05)

for i, cls in enumerate(classes):
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    
    # Take the first image
    img_path = os.path.join(class_path, images[0])
    img = np.array(Image.open(img_path))
    
    # Separate RGB channels
    r_channel = img.copy()
    r_channel[:, :, 1] = 0  # Zero out G
    r_channel[:, :, 2] = 0  # Zero out B
    
    g_channel = img.copy()
    g_channel[:, :, 0] = 0  # Zero out R
    g_channel[:, :, 2] = 0  # Zero out B
    
    b_channel = img.copy()
    b_channel[:, :, 0] = 0  # Zero out R
    b_channel[:, :, 1] = 0  # Zero out G
    
    # Plot original and channels
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"{cls}\nOriginal", fontsize=10)
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(r_channel)
    axes[i, 1].set_title(f"{cls}\nRed Channel", fontsize=10)
    axes[i, 1].axis("off")
    
    axes[i, 2].imshow(g_channel)
    axes[i, 2].set_title(f"{cls}\nGreen Channel", fontsize=10)
    axes[i, 2].axis("off")
    
    axes[i, 3].imshow(b_channel)
    axes[i, 3].set_title(f"{cls}\nBlue Channel", fontsize=10)
    axes[i, 3].axis("off")

plt.tight_layout()
plt.show()

```


    
![png](output_files/output_10_0.png)
    



```python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the dataset
data_dir = os.path.join("data")
classes = ["cloudy", "desert", "green_area", "water"]
output_dir = "visuals"
# Set up the plot
fig, axes = plt.subplots(len(classes), 3, figsize=(12, 8))  # 3 columns: R, G, B histograms
plt.suptitle("Color Histogram Analysis of Sample Images by Category", fontsize=16, y=1.05)

for i, cls in enumerate(classes):
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    
    # Take the first image
    img_path = os.path.join(class_path, images[0])
    img = np.array(Image.open(img_path))
    
    # Compute histograms for each channel
    r_hist = np.histogram(img[:, :, 0], bins=256, range=(0, 256))[0]  # Red channel
    g_hist = np.histogram(img[:, :, 1], bins=256, range=(0, 256))[0]  # Green channel
    b_hist = np.histogram(img[:, :, 2], bins=256, range=(0, 256))[0]  # Blue channel
    
    # Plot histograms
    axes[i, 0].plot(r_hist, color="red", label="Red")
    axes[i, 0].set_title(f"{cls}\nRed Channel", fontsize=10)
    axes[i, 0].set_xlim(0, 256)
    axes[i, 0].set_ylim(0, np.max([r_hist, g_hist, b_hist]) * 1.1)  # Adjust y-axis for visibility
    axes[i, 0].grid(True, alpha=0.3)
    
    axes[i, 1].plot(g_hist, color="green", label="Green")
    axes[i, 1].set_title(f"{cls}\nGreen Channel", fontsize=10)
    axes[i, 1].set_xlim(0, 256)
    axes[i, 1].set_ylim(0, np.max([r_hist, g_hist, b_hist]) * 1.1)
    axes[i, 1].grid(True, alpha=0.3)
    
    axes[i, 2].plot(b_hist, color="blue", label="Blue")
    axes[i, 2].set_title(f"{cls}\nBlue Channel", fontsize=10)
    axes[i, 2].set_xlim(0, 256)
    axes[i, 2].set_ylim(0, np.max([r_hist, g_hist, b_hist]) * 1.1)
    axes[i, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "colour_analysis.jpg"), format="jpg", dpi=300, bbox_inches="tight")
plt.show()

```


    
![png](output_files/output_11_0.png)
    



```python
pip install scikit-learn
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting scikit-learn
      Downloading scikit_learn-1.6.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
    Requirement already satisfied: numpy>=1.19.5 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in /home/rjaswal1634/.local/lib/python3.12/site-packages (from scikit-learn) (1.15.2)
    Collecting joblib>=1.2.0 (from scikit-learn)
      Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting threadpoolctl>=3.1.0 (from scikit-learn)
      Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
    Downloading scikit_learn-1.6.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.1 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.1/13.1 MB[0m [31m26.5 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m0:01[0m:01[0m
    [?25hDownloading joblib-1.4.2-py3-none-any.whl (301 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m301.8/301.8 kB[0m [31m37.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
    Installing collected packages: threadpoolctl, joblib, scikit-learn
    Successfully installed joblib-1.4.2 scikit-learn-1.6.1 threadpoolctl-3.6.0
    Note: you may need to restart the kernel to use updated packages.



```python
import os
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

# Path to the dataset
data_dir = os.path.join("data")  # Adjust if folder name differs
output_dir = os.path.join("processed_data")  # Where to save train/val/test
classes = ["cloudy", "desert", "green_area", "water"]

# Create output directories
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Load and split data
for cls in classes:
    images = os.listdir(os.path.join(data_dir, cls))
    images = [img for img in images if img.endswith(".jpg")]  # Filter for .jpg files
    train_val, test = train_test_split(images, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 of 90% = 10% of total

    # Copy and resize images
    for split, split_data in [("train", train), ("val", val), ("test", test)]:
        for img in split_data:
            try:
                img_path = os.path.join(data_dir, cls, img)
                img_out = os.path.join(output_dir, split, cls, img)
                with Image.open(img_path) as im:
                    im.resize((224, 224)).save(img_out)
            except Exception as e:
                print(f"Error processing {img}: {e}")

print("Data preparation complete!")
```

    Data preparation complete!



```python
# Verify the new structure
for split in ["train", "val", "test"]:
    print(f"\nSplit: {split}")
    for cls in classes:
        class_path = os.path.join(output_dir, split, cls)
        num_images = len(os.listdir(class_path))
        print(f"{cls}: {num_images} images")
```

    
    Split: train
    cloudy: 1200 images
    desert: 1373 images
    green_area: 1200 images
    water: 1200 images
    
    Split: val
    cloudy: 150 images
    desert: 254 images
    green_area: 150 images
    water: 150 images
    
    Split: test
    cloudy: 150 images
    desert: 249 images
    green_area: 150 images
    water: 150 images



```python

```
