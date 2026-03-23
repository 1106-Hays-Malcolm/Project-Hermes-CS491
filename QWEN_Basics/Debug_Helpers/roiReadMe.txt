roiTest.py 

This is a simple debugging program using command line argument to identify the current ROI in a specific qwen version

commandline:

python roiTest.py qwen_screenshot_V3_1.py

expected output will be:
PS C:\LLM_Project\QWEN> python roiTest.py qwen_screenshot_V2_2.py
left = 500
top = 500
width = 100
height = 100

along with a saved roiTest.png that shows exactly what is being tested

Tells you the region and gives in image so you can figure out the ROI issues without running the LLM everytime
