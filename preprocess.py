import os
import cv2
import glob
import imutils
import numpy as np

OUT_PATH = 'D:\\Data\\Retinop\\PreProcessed\\train_images\\'


def main():

    retinopFiles = glob.glob('D:\\Data\\Retinop\\train_images\\*.png')
    print("Number of Retina Images: ", len(retinopFiles))
    for retina in retinopFiles:
        img = cv2.imread(retina)
        print("Retina Image Name: {} -- Image Size: {} ".format(os.path.basename(retina), img.shape))
        if img.shape[0] > 720:
            scale = 720 / img.shape[0]
            dim = (int(scale * img.shape[1]), 720)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # print(img.shape)
        # cv2.imshow("Original Image", img)
        # cv2.waitKey(2000)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

        # Now find contours in it. There will be only one object, so find bounding rectangle for it.
        # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        # determine the most extreme points along the contour
        l = tuple(c[c[:, :, 0].argmin()][0])
        r = tuple(c[c[:, :, 0].argmax()][0])
        t = tuple(c[c[:, :, 1].argmin()][0])
        b = tuple(c[c[:, :, 1].argmax()][0])
        print(l, r, t, b)
        x = l[0]
        y = t[1]
        w = r[0] - l[0]
        h = b[1] - t[1]

        if w > h:
            dw2 = int((w - h) / 2)
            x += dw2
            w -= 2 * dw2
        elif w < h:
            dh2 = int((h - w) / 2)
            y += dh2
            h -= 2 * dh2
 
        # Now crop the image, and save it into another file.
        crop = img[y:y+h, x:x+w]

        width = 320
        height = 320
        dim = (width, height)
        # resize image
        resized = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)

        # apply CLAHE to color images - convert to LAB format and apply to lightness - then convert back to BGR
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


        # create a CLAHE object (Arguments are optional) - for gray scale images
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # contrast = clahe.apply(resized)

        if cv2.imwrite(OUT_PATH + os.path.basename(retina), contrast):
            print('PreProcessed Retina file {} saved to disk...'.format(retina))
        else:
            print('PreProcessed Retina file {} NOT saved to disk...'.format(retina))

        # print('Contrast: ', np.amin(contrast), np.amax(contrast))
        # print('Contrast Enhanced Dimensions : ', contrast.shape)
        # cv2.imshow("Constrast Enhanced image", contrast)
        # cv2.waitKey(2000)

        cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()