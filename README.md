# Braille-to-Text Converter

This Python script converts images containing Braille characters into readable text. The program processes the Braille image and extracts the textual information.

## Overview

Braille is a tactile writing system used by people with visual impairments. This script is designed to interpret Braille patterns in images and translate them into corresponding English characters.

The script undergoes several image processing steps to transform the Braille symbols into readable text. The primary functionalities include:

- **Image Preprocessing:**
  - **Dilation and Erosion:** Initial image operations to ensure optimal dot connectivity and remove noise.
  - **Connected Component Analysis (CCA):** Identifying and labeling distinct components (Braille dots) within the image.

- **Pattern Matching:**
  - A predefined key containing Braille symbols is used to match individual Braille characters to their corresponding English characters.

## Features

### Preprocessing Steps

The preprocessing steps include:
- **Dilation and Erosion:** Used to adjust the size and shape of Braille dots for better pattern recognition.
- **Connected Component Analysis:** Identifies and labels individual Braille characters within the image.

### Pattern Matching

- **Braille Key Matching:** The script matches Braille characters in the input image with the characters provided in the Braille key to obtain their English counterparts.
- **Textual Output:** Produces a text output representing the English equivalent of the input Braille characters.

## Usage

### Prerequisites

- **Libraries:** Ensure you have the required libraries installed:
  - `cv2`
  - `numpy`

### Usage Instructions

1. **Input Images:**
   - Provide the Braille image to be converted. This image should contain Braille characters or words.
   - A key image with Braille symbols is required to decode the characters.

2. **Running the Script:**
   - Run the Python script after setting the file paths for the Braille and key images.
   - The script performs image processing and matches the Braille characters with their corresponding English text.

---
*Documented by ChatGPT*
