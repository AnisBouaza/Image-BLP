import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import patches as patches

def calc_mse(a, b):
    sum = 0.0
    for i in range(256):
        sum = sum + pow((a[i][0] - b[i][0]), 2)
    return sum / 256

def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        pass
      
    return new_value 
   
def lbp_calculated_pixel(img, x, y):
   
    center = img[x][y] 
   
    val_ar = [] 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
    val_ar.append(get_pixel(img, center, x-1, y)) 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1)) 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 
   
path = 'img.png'
img_bgr = cv2.imread(path, 1) 
   
height, width, _ = img_bgr.shape 
   
#conversion gris 
img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) 
   
# Crée une numpy array avec meme longeur et largeur que l'image
img_lbp = np.pad(img_gray, 0, mode="edge")
   
for i in range(0, height):
    for j in range(0, width): 
        img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 
  

f1 = plt.figure(1,figsize=(8, 3))
plt.subplot(1,2,1)
plt.title("Image origine")
plt.imshow(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Image LBP")
plt.imshow(img_lbp, cmap ="gray")




f2 = plt.figure(2,figsize=(10, 3))
x = np.random.randint(0, width-64)
y = np.random.randint(0, height-64)
width_box = 64
height_box = 64

plt.subplot(1,3,1)
plt.title("Encadrement d'une zone")
plt.imshow(img_lbp, cmap ="gray")
ax = plt.gca()
rect = patches.Rectangle((x,y),
                 width_box,
                 height_box,
                 linewidth=2,
                 edgecolor='red',
                 fill = False)

ax.add_patch(rect)

plt.subplot(1,3,2)
plt.title("Zoom sur la coupe")
img_crop = img_lbp[y:y+height_box, x:x+width_box]
plt.imshow(img_crop, cmap ="gray") 


plt.subplot(1,3,3)
plt.title("Histogramme")
histo_ref = cv2.calcHist([img_crop], [0], None, [256], [0, 256])
plt.hist(img_crop.ravel(),256,[0,256])





  

cv2.imshow('image', img_lbp) 
# mouse_click. 
def mouse_click(event, x, y,  
                flags, param): 
      
    if event == cv2.EVENT_LBUTTONDOWN:
            f3 = plt.figure(3,figsize=(10, 3))
            width_box = 64
            height_box = 64

            plt.subplot(1,3,1)
            plt.title("Encadrement de la zone séléctioné")
            plt.imshow(img_lbp, cmap ="gray")
            ax = plt.gca()
            rect = patches.Rectangle((x,y),
                            width_box,
                            height_box,
                            linewidth=2,
                            edgecolor='red',
                            fill = False)

            ax.add_patch(rect)

            plt.subplot(1,3,2)
            plt.title("Zoom sur la coupe")
            img_crop = img_lbp[y:y+height_box, x:x+width_box]
            histo_selected = cv2.calcHist([img_crop], [0], None, [256], [0, 256])

            plt.imshow(img_crop, cmap ="gray") 
            plt.subplot(1,3,3)
            plt.title("Histogramme")
            print("Resultat Comparaison : ", calc_mse(histo_selected, histo_ref))
            plt.hist(img_crop.ravel(),256,[0,256])
            f3.canvas.manager.window.move(500,500)
            plt.show()

cv2.setMouseCallback('image', mouse_click) 
   
f1.canvas.manager.window.move(0,0)
f2.canvas.manager.window.move(0,500)
plt.show()

cv2.waitKey(0) 
cv2.destroyAllWindows() 










