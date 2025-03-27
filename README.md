<div align="center" markdown> 

<img src=""/>

# Merge Multispectral Images
  
<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#examples">Examples</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/merge-multispectral)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/merge-multispectral.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/merge-multispectral.png)](https://supervisely.com)

</div>

## Overview

This application helps you merge multiple channels of multispectral images into a single RGB image. It's particularly useful when you have separate images for different spectral bands (like red, green, or blue channels) and want to combine them into a single color image.

**Notes**

- The original project and images remain unchanged
- Make sure all channel images have the same dimensions
- The app will skip groups where any required channel is missing

## Features

- **Channel Order Control**: Define the order of channels in the output RGB image
- **Annotation Preservation**: All labels and annotations are preserved in the merged image
- **Project Structure**: Maintains the original dataset hierarchy

## How to Use

### Step 1: Launch the App

1. Open the app from the Supervisely ecosystem / Context menu of project / Context menu of dataset
2. Configure the channel order in the modal window

### Step 2: Configure Channel Order

In the modal window, you need to specify the order of channels in YAML format:
```yaml
R: _0  # Red channel (replace _0 with your red channel suffix)
G: _1  # Green channel (replace _1 with your green channel suffix)
B: _2  # Blue channel (replace _2 with your blue channel suffix)
```

Replace the suffixes (_0, _1, _2) with the actual suffixes used in your image filenames.

### Step 3: Run the App

1. Click "Run" to start the merging process
2. Monitor the progress in the logs
3. Wait for the process to complete

### Step 4: Access Results

- A new project will be created with the prefix "Merged multispectral"
- The new project maintains the same structure as the source project
- Each merged image will be saved as a PNG file
- All annotations are preserved and attached to the merged images

## Examples

### Input Images

Your source images might look like this:
- `image_0.png` (Red channel)
- `image_1.png` (Green channel)
- `image_2.png` (Blue channel)

### Output
The app will create a single RGB image combining all channels:
- `image.png` (Combined RGB image)

### Channel Order Example

If your images are named like this:

```text
image_red.png
image_green.png
image_blue.png
```

Your YAML configuration should be:

```yaml
R: _red
G: _green
B: _blue
```
