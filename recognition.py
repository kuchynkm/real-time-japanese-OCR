"""Real-time hiragana character recognition script

It performs real-time OCR on video/camera input to recognize Japanese hiragana characters. 
The OCR is realized by a convolutional neural network.

Input:
------
	image from camera input

Output:
-------
	image with character prediction

Usage:
------
	After runing the script, a window with camera input opens.
	In the center of the window, there is a gray rectangle specifying the recognition region. 
	To perform OCR on a given character, aim the camera on the character so that the 
	character is contained within the recognition region. 
	The script automatically performs inference on the content of the recognition region and 
	prints the result on top of the recognition region rectangle.
	In the upper-right corner of the window, a preview of the actual image input for the neural network is showing.
	Close the window by pressing Esc.

Dependencies:
	tensorflow 
	numpy
	cv2
	pillow

"""


#-----------------------------------------------------------------
# Importing libraries
#-----------------------------------------------------------------

from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image          
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import sys
import argparse


#----------------------------------------------------------------
# Some heplful functions
#----------------------------------------------------------------

# Function that finds contours in the image and returns them in a list
def get_contours(img, threshold, invert = True):
	# Convert img to gray scale
	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
	except:
		gray = img
	
	# Invert or not to invert the image?
	if invert:
		bin_type = cv2.THRESH_BINARY_INV
	else:
		bin_type = cv2.THRESH_BINARY
		
	# Applying threshold to get {0,255} array
	retval, thresh_gray = cv2.threshold(gray, thresh=threshold*255, maxval=255, type=bin_type) 

	# Get the objects contours
	contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours


# Detection of boundary of a character based on its contours
def single_char_boundary(img, contours):
	# Minimal area for a contour to be still considered a part of the character
	min_area = 0 
	# Initialize upper and lower bounds for rectangle corner coordinates
	min_x = img.shape[1]
	min_y = img.shape[0]
	max_x, max_y =  0, 0

	# Get the upper and lower bounds on coordinates enclosing the character
	for cont in contours:
		x,y,w,h = cv2.boundingRect(cont)
		if w*h >= min_area:
			if x < min_x:
				min_x = x
			if y < min_y:
				min_y = y
			if x + w > max_x:
				max_x = x + w
			if y + h > max_y:
				max_y = y + h
				
	return min_x, min_y, max_x, max_y

# Normalize the image to [0,255] for opencv
def normalize(img_array, max_val = 255):
	img = (max_val/np.amax(img_array))*img_array
	img = img.astype(np.uint8)
	return img


# Normalize and crop
def char_crop(img_array, crop_threshold, invert = False):
	#img_copy = img_array.copy()

	# Normalize
	img = normalize(img_array)

	# Erode
	kernel = np.ones((2,2), np.uint8)
	eroded_img = cv2.erode(img, kernel, iterations = 1)
	
	# Get contours of objects
	contours = get_contours(eroded_img, threshold = crop_threshold, invert = invert)
	
	# Find the biggest bounding box based on the objects contours
	x1, y1, x2, y2 = single_char_boundary(img, contours)
	
	# Crop the image to bounding box (if not empty)
	if x1 < x2 and y1 < y2:
		crop_image = img[y1:y2, x1:x2]
	else:
		crop_image = img
	
	return crop_image


'''
Preprocessing image array - normalization and conversion to array 
of strictly zeros and ones based on given threshold
''' 
def preprocess_array(img, target_size, threshold, crop_threshold, invert = True):
	# Convert to grayscale
	try:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	except:
		pass

	# Crop the character
	crop_img = char_crop(img, crop_threshold = crop_threshold, invert = invert) # crop_threshold = 0.25 for training, else 0.5

	# Resize
	img = cv2.resize(crop_img, target_size, interpolation = cv2.INTER_AREA)
	
	# Invert or not to invert the image? (Invert if dark background and light character)
	if invert:
		bin_type = cv2.THRESH_BINARY_INV
	else:
		bin_type = cv2.THRESH_BINARY
		
	# Separate character from background by setting threshold   
	_, img = cv2.threshold(img, thresh=threshold*255, maxval=255, type = bin_type)
	
	# Normalize to [0,1] (avoiding division by zero when image black)
	if np.amax(img) > 0:
		img = (1/np.amax(img))*img
	else:
		img = img*0
	
	return img


# Predict character in image
def predict(img, threshold, crop_threshold, invert = True, render = True):
	# Preprocess the image for the model
	x = preprocess_array(img, target_size = (40,40), threshold = threshold, crop_threshold = crop_threshold, invert = invert)
	global W, outputImage

	# Render preview of the model input in the upper-right corner
	if render:
		preview_img = cv2.resize(x, (100,100), interpolation = cv2.INTER_AREA)
		preview_img = cv2.cvtColor(np.float32(255*preview_img),cv2.COLOR_GRAY2RGB)
		outputImage[0:100, W-100:W] = preview_img

	# Format the image array to fit the model's input shape
	x = np.expand_dims(x, axis = 0)
	x = np.expand_dims(x, axis = -1)
	
	# Run prediction
	y = model.predict(x)
	return np.argmax(y), np.amax(y)


# Put text on BGR image
def put_text(img, text, position, font, size, color):
	# Convert to RGB for pillow
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Create pillow image
	pil_img = Image.fromarray(rgb_img)

	# Put text on image using pillow
	draw = ImageDraw.Draw(pil_img)
	try:
		font = ImageFont.truetype(font, size)
	except:
		sys.exit('Font ' + str(font) + ' not found.')

	draw.text(position, text, font=font, fill=color + (0,))

	return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Callback for recognition region size slider
def reg_size_callback(val):
	global min_x, min_y, max_x, max_y
	a = int(val)
	min_x = W_C - (min_a + a)
	min_y = H_C - (min_a + a)
	max_x = W_C + (min_a + a)
	max_y = H_C + (min_a + a)

# Callback for threshold slider
def threshold_callback(val):
	global threshold
	threshold = val/255.

# Callback for threshold slider
def crop_threshold_callback(val):
	global crop_threshold
	crop_threshold = val/255.

# Callback for invert slider
def invert_callback(val):
	global invert
	invert = bool(val)



#----------------------------------------------------------------
# Main
#----------------------------------------------------------------
if __name__ == "__main__":

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', action='store', type=str)
	parser.add_argument('--source', action='store', type=str)
	args = parser.parse_args()
	try:
		input_type = args.input
		source = args.source
	except:
		pass


	# Load the model
	try:
		model = load_model('model/hiragana_model_acc9914.h5')
		print('Model loaded succesfully.')
	except:
		sys.exit('Model not found.')


	# Hiragana dictionary for translation of model predictions
	ind_to_char = {0: 'あ', 1: 'い', 2: 'う', 3: 'え', 4: 'お', 5: 'か', 6: 'が', 7: 'き', 8: 'ぎ', 9: 'く', 10: 'ぐ', 
					11: 'け', 12: 'げ', 13: 'こ', 14: 'ご', 15: 'さ', 16: 'ざ', 17: 'し', 18: 'じ', 19: 'す', 20: 'ず', 
					21: 'せ', 22: 'ぜ', 23: 'そ', 24: 'ぞ', 25: 'た', 26: 'だ', 27: 'ち', 28: 'ぢ', 29: 'つ', 30: 'づ', 
					31: 'て', 32: 'で', 33: 'と', 34: 'ど', 35: 'な', 36: 'に', 37: 'ぬ', 38: 'ね', 39: 'の', 40: 'は', 
					41: 'ば', 42: 'ぱ', 43: 'ひ', 44: 'び', 45: 'ぴ', 46: 'ふ', 47: 'ぶ', 48: 'ぷ', 49: 'へ', 50: 'べ', 
					51: 'ぺ', 52: 'ほ', 53: 'ぼ', 54: 'ぽ', 55: 'ま', 56: 'み', 57: 'む', 58: 'め', 59: 'も', 60: 'や', 
					61: 'ゆ', 62: 'よ', 63: 'ら', 64: 'り', 65: 'る', 66: 'れ', 67: 'ろ', 68: 'わ', 69: 'を', 70: 'ん'}


	# Initialize window for camera input and chose the camera device
	window_name = 'Real-time Character Recognition'
	cv2.namedWindow(window_name, flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
	
	if input_type == 'webcam':
		try:
			cap = WebcamVideoStream(src = int(source)).start()
			print('Webcam stream started.')
		except:
			sys.exit('Video stream failed.')
	elif  input_type == 'ipcam':
		try:
			cap = WebcamVideoStream(src = source).start()
			print('IP camera stream started.')
		except:
			sys.exit('Video stream failed.')
	else:
		try:
			cap = WebcamVideoStream(src = 0).start()
			print('Video stream started.')
		except:
			sys.exit('Video stream failed.')


	# Get the camera input shape
	try:
		Input = cap.read()
		H = Input.shape[0]
		W = Input.shape[1]
	except:
		H = 480
		W = 640
		

	# Set coordinates of the center of the image and corners of the recognition area
	H_C = int(H/2)
	W_C = int(W/2)
	min_a = 5
	a = int(H/9)
	min_x = W_C - (min_a + a)
	min_y = H_C - (min_a + a)
	max_x = W_C + (min_a + a)
	max_y = H_C + (min_a + a)

	# Set threshold and cropping threshold values:
	threshold = 0.6
	crop_threshold = 0.5
	invert = True

	# Set trackbars
	cv2.createTrackbar('Region size', 'Real-time Character Recognition', a, int(min(H/2,W/2)), reg_size_callback)
	cv2.createTrackbar('Preprocessing threshold', 'Real-time Character Recognition', int(threshold*255), 255, threshold_callback)
	cv2.createTrackbar('Cropping threshold', 'Real-time Character Recognition', int(crop_threshold*255), 255, crop_threshold_callback)
	cv2.createTrackbar('Invert', 'Real-time Character Recognition', int(invert), 1, invert_callback)

 
	# Initialize prediction text with 'Unknown'  
	text = 'Unknown'

	# Initialize FPS
	fps = FPS().start()

	while True:
		imgInput = cap.read()    
		outputImage = imgInput.copy()
		
		# Recognition will be performed only within this rectangle in the center
		crop_img = outputImage[min_y:max_y, min_x:max_x]                           
				  
		# For dark character and light background: invert = True, else: invert = False
		prediction = predict(crop_img, threshold = threshold, crop_threshold = crop_threshold, invert = invert, render = True) 

		# Print only prediction with high certainty and filter false positives
		if prediction[1] >= 0.9:
			text = u'{character} {percent:1.0f}%'.format(character=ind_to_char[prediction[0]], percent=100*prediction[1])
		elif prediction[1] < 0.5:
			text = 'Unknown'
		
		# Print prediction
		cv2.rectangle(outputImage, (min_x, min_y), (max_x, max_y), (100,100,100), 1)
		outputImage = put_text(outputImage, text, (min_x, min_y - 20), 'BIZ-UDGothicB.ttc', 15, (100,100,100))
		
		# Show the image with prediction
		cv2.imshow(window_name, outputImage)

		# Update FPS
		fps.update()

		# Wait for Esc to stop the loop
		if cv2.waitKey(1) & 0xFF == 27:
			break

	# Stop FPS and print approximate value
	fps.stop()
	print("Approximate FPS: {:.2f}".format(fps.fps()))

	# Close the window and release camera
	cv2.destroyAllWindows()
	cap.stop()
