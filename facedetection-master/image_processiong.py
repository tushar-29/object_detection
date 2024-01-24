# importing OpenCV Lib
import cv2

# # access of web cam o the device
# webcam = cv2.VideoCapture(0)
#
# # reading each frame image from the web cam
# _, img = webcam.read()
#
# # looping the n dimension numpy RGB array
# for i in range(len(img)):
#     for j in range(len(img[i])):
#         # separating to RGB values
#         R, G, B = img[i][j]
#         # grayscale conversion formula
#         img[i][j] = [0.299 * R + 0.59 * G + 0.11 * B]
#
# # saving the gray scale image
# cv2.imwrite('new_img_1.png', img)



# cv2.imwrite("test_img.png", img)


# webcam.release()
# cv2.destroyAllWindows()

import numpy as np

img = cv2.imread("new_img_1.png")
mod_img = np.insert(img, 0, 0, axis=0)
mod_img = np.insert(mod_img, 0, 0, axis=1)

height = len(img)
width = len(img[0])
integral_arr = np.array([])

for i in range(1, height):
    sub_arr = np.array([])
    for j in range(1, width):
        sum = 0
        for x in range(0, i + 1):
            for y in range(0, j + 1):
                sum += mod_img[x][y][0]
        sub_arr = np.append(sub_arr, sum)
    integral_arr = np.append(integral_arr, sub_arr)


arr = [[1, 2, 4, 1],
       [6, 7, 3, 5],
       [8, 2, 1, 6],
       [3, 1, 3, 7]]

arr.insert(0, [0, 0, 0, 0])
for i in arr:
    i.insert(0, 0)


print("new arr = ", arr)

print(arr)
in_arr = []
sum = 0

for i in range(1, len(arr)):
    sub_arr = []
    for j in range(1, len(arr[i])):
        for x in range(0, i+1):
            for y in range(0, j+1):
                sum += arr[x][y]
                print(arr[x][y], end="\t")
        sub_arr.append(sum)
        print("\nsum = ", sum)
        sum = 0
    print("sub_arr = ", sub_arr)
    in_arr.append(sub_arr)

print("INTEGRAL ARR : ")
print(in_arr)