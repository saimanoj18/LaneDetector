
# coding: utf-8

# In[56]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip 
from IPython.display import HTML

get_ipython().magic('matplotlib inline')


# In[57]:


#change the BGR image to grayscale 
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#perform canny edge detection
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

#perform Gaussian Blur
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# In[58]:


#ROI detection based on the polygon created using vertices
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# In[59]:


#This is where the actual changes lies from the template

#Things considered:
# Lets divide the image into two halfs, left half and right half
#it was found during experiments that sometimes in left half, some line segments show up that have an opposite sloe compared to all the other segments in the same half.
#Hence, we remove the line segments that have opposite slope than what is expected from the left half
#In short, we remove the line segments that have different slope from than the expected slope (positive or negative depending on left or right half)
#The expected thing is that the line segments in the left half should have negative slope while the line segments in the right half should have postive slop

def draw_lines_modified (line_img, lines,color=[255, 0, 0], thickness=4):
    count = 0 
    count1 = 0
    left_x1 = lefty1 = left_x2 = left_y2 = 0
    right_x1 = right_x2 = right_y1 = right_y2 = 0
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
                
                slope = (y2-y1)/(x2-x1)
                #print(slope)
                if (slope < 0):# we check if the slope is negative, if negative, it should be in left half
                    if (x2 > 500):#checking if the line segments lie in right half, if yes, reject them 
                        continue
                    if (x1 > 500):#checking if the line segments lie in right half, if yes, dont consider them
                        continue
                    #print(x1, y1, x2, y2)
                    #cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
                    
                    if (count == 0):
                        left_x1 = x1
                        left_y1 = y1
                        left_x2 = x2
                        left_y2 = y2
                        count = count + 1
                        
                    else:
                        if (x1 < left_x1):#taking the end values
                            left_x1 = x1
                            left_y1 = y1
                        if (x2 > left_x2):
                            left_x2 = x2
                            left_y2 = y2
                    
                elif (slope > 0):# we check if the slope is postive, if yes, it should be in right half
                    if (x2 < 500):
                        continue
                    if (x1 < 500):# rejecting the ones that are not in the right half
                        continue
                    #cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
                    #print(x1, y1, x2, y2)
                    #cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
                    
                    if (count1 == 0):
                        right_x1 = x1
                        right_y1 = y1
                        right_x2 = x2
                        right_y2 = y2
                        count1 = count1 + 1
                        
                    else:#finding the lend tips of all line segments
                        if (x1 < right_x1):
                            right_x1 = x1
                            right_y1 = y1
                        if (x2 > right_x2):
                            right_x2 = x2
                            right_y2 = y2
                    

                else:
                    print("first pass done...")

    slope_left = (left_y2-left_y1)/(left_x2-left_x1)# estimating line slope from the found line tips in the left half
    c_left = left_y2-(slope_left*left_x2)
    new_left_y3 = 539;#y=mx+c
    new_left_x3 = (int)(new_left_y3-c_left)/slope_left#from the line equation, estimate the end points
    new_left_x3 = int(new_left_x3)
   
    
    new_left_y4 = 330;#y=mx+c
    new_left_x4 = (int)(new_left_y4-c_left)/slope_left
    new_left_x4 = int(new_left_x4)  #from the line equation, estimate the end points

    cv2.line(line_img, (new_left_x3, new_left_y3), (new_left_x4, new_left_y4), color, thickness)#draw estimated line
    
    slope_right = (right_y2-right_y1)/(right_x2-right_x1)# estimating line slope from the found line tips in the right half
    c_right = right_y2-(slope_right*right_x2)
    new_right_y3 = 539;#y=mx+c
    new_right_x3 = (int)(new_right_y3-c_right)/slope_right #from the line equation, estimate the end points
    new_right_x3 = int(new_right_x3)


   
    new_right_y4 = 330;#y=mx+c
    new_right_x4 = (int)(new_right_y4-c_right)/slope_right
    new_right_x4 = int(new_right_x4) #from the line equation, estimate the end points
    #print (new_right_x3, new_right_y3, right_x2, right_y2)


    cv2.line(line_img, (new_right_x3, new_right_y3), (new_right_x4, new_right_y4), color, thickness)#draw estimated lines


    


# In[60]:


def hough_lines_mod(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_modified(line_img, lines)
    return line_img


# In[61]:


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# In[62]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    gray_image = grayscale(image)

    blur_image = gaussian_blur (gray_image, 7)

    canny_image = canny(blur_image, 70, 140)

    points = np.array([[60, 539], [400, 320], [600, 320], [940, 539]])

    masked_image = region_of_interest(canny_image,np.int32([points]) )
    
    hough_image = hough_lines_mod(masked_image, 2, 1, 15, 20, 80)
    
    lanes_on_image = weighted_img(hough_image, image, 0.6, 5, 0)



    return lanes_on_image


# In[65]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
#white_output = 'test_videos_output/solidYellowLeft.mp4'


## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)




clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")


white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')




# In[66]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




