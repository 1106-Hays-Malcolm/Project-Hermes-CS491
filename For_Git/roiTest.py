import sys
import mss
import mss.tools

def getROI(filename):
    in_region = False
    roi_define = []
    left = 0
    top = 0
    width = 0
    height = 0

    with open(filename, "r") as file:
        for line in file:
            if "REGION" in line:
                in_region = True

            if in_region:
                roi_define.append(line)

                if "}" in line:
                    in_region = False
                    break

    for line in roi_define:
        if '"left"' in line:
            left = int(line.split(":")[1].strip().rstrip(","))
        elif '"top"' in line:
            top = int(line.split(":")[1].strip().rstrip(","))
        elif '"width"' in line:
            width = int(line.split(":")[1].strip().rstrip(","))
        elif '"height"' in line:
            height = int(line.split(":")[1].strip().rstrip(","))

    return left, top, width, height


def saveROIimage(left, top, width, height):
    region = {
        "left": left,
        "top": top,
        "width": width,
        "height": height
    }

    with mss.mss() as sct:
        sct_img = sct.grab(region)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output="roiTest.png")


filename = sys.argv[1]

left, top, width, height = getROI(filename)

print(f"left = {left}")
print(f"top = {top}")
print(f"width = {width}")
print(f"height = {height}")

saveROIimage(left, top, width, height)

print("Saved roiTest.png")