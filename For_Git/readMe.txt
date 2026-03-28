I have several redundant files here as I was running these locally while testing/comparing GPU usage.

The test_image.png is an image of BDG3 coordinate close to correct size for testing inference.


The naming convention is a little unorthodox and we can change it later I just needed to keep track of which ones were doing what.

Each has a name such as qwen_screenShot_V(x)_(y)

The x value identifies what precision value is being used
1 - 16BFloat the full 7B parameters
2 - 8bit quantized version
3 - 4bit quantized version

The y value identifies Where it is loading in from and if it has a timer feature
Read Additional Notes before running y =3 or 4
1 - loads  in from Hugging Face no timer
2 - loads in from Hugging Face with timer
3 - loads in locally no timer
4 - loads in locally with timer

Example: qwen_screenShot_V2_4 - 8 bit version loads local and has timer 



Before running any versions that load locally do this:

Locate Hugging Face Model Cache (Windows)

After downloading the model, it will be stored in the Hugging Face cache directory:

C:\Users\<username>\.cache\huggingface\hub

Inside this folder, locate the model directory and copy the contents of the latest `snapshots` folder into your local model directory. ...snapshots\cc59....\(these are the files to copy into your directory)

Around line 17 locate MODEL_NAME and update the file path to where you have this saved.

I think that is about it. Holler if you have any questions














qwen_screenshot_v1

This version loads the full QWEN2.5-VL-7B model.

Overview

1. Capture screen region using MSS
2. Convert image to PIL format
3. Send image + prompt to Qwen model
4. Model returns text (X:Y format)
5. Regex extracts values
6. Output formatted as JSON

Currently a fixed ROI this will change in later versions

Output 
JSON
{"x":"<value>","y":"<value>"}

## Hardware / Device Behavior

This project attempts to run the model on a CUDA-enabled GPU when available. However, due to the size of the **Qwen2.5-VL-7B** model, the GPU may not have sufficient VRAM to hold the entire model.

When this occurs, the Transformers library automatically offloads portions of the model to the CPU using:
- `device_map="auto"`

### What This Means
- Some model layers run on the GPU
- Some layers are offloaded to the CPU
- Temporary "meta" device placeholders may be used during loading

### Impact
- Inference will still run correctly
- Performance will be slower compared to full GPU execution
- Behavior is expected and not an error

### Console Message Example
------------------------------------------------------------------------------
qwen_screenShot_V2

## 8-Bit Quantized Version

This version uses **8-bit quantization** via `bitsandbytes` to reduce GPU memory usage while maintaining the same coordinate extraction pipeline.

### Key Changes
- Model is loaded using:
  - `BitsAndBytesConfig(load_in_8bit=True)`
- `device_map="auto"` is used for automatic placement on GPU
- No manual dtype configuration is required

### Benefits
- Significantly reduced VRAM usage
- Higher likelihood of fitting the full model on GPU
- Maintains same output format and workflow as V1

### Expected Behavior
- Model may still offload small portions to CPU depending on available VRAM
- Minor warnings during load (e.g., casting or processor changes) are expected and do not affect functionality
---------------------------------------------------------------
qwen_screenShot_V3

## 8-Bit Quantized Version

This version uses **4-bit quantization** via `bitsandbytes` to reduce GPU memory usage while maintaining the same coordinate extraction pipeline.

### Key Changes
- Model is loaded using:
  - `BitsAndBytesConfig(load_in_8bit=True)`
- `device_map="auto"` is used for automatic placement on GPU
- No manual dtype configuration is required

### Benefits
- Significantly reduced VRAM usage
- Higher likelihood of fitting the full model on GPU
- Maintains same output format and workflow as V1

### Expected Behavior
- Model may still offload small portions to CPU depending on available VRAM
- Minor warnings during load (e.g., casting or processor changes) are expected and do not affect functionality
-----------------------------------------------------------------------------------


GitHub push on 3/23/26

I made a mistake when cleaning up the code prior to the last push. 

1 There was argument error in the capture_region function. This is now fixed

2 I removed the line that saves a png of the screenshot. It was useful for making sure I have the correct ROI so I added it back in.

3 I have another small helper py I made included. It uses a commandline argument of a py file that prints the current ROI from that py file

example in command line

python roiTest.py qwen_screenshot_V3_1.py

expected output will be:
PS C:\LLM_Project\QWEN> python roiTest.py qwen_screenshot_V2_2.py
left = 500
top = 500
width = 100
height = 100
Saved roiTest.png

Tells you the region and gives in image so you can figure out the ROI without running the LLM everytime

