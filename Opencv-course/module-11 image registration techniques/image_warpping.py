#image warper and homography
img1n=cv2.imread("pano1.jpg",cv2.IMREAD_COLOR)
img2n=cv2.imread("pano2.jpg",cv2.IMREAD_COLOR)

img1=cv2.cvtColor(img1n,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2n,cv2.COLOR_BGR2GRAY)

orb=cv2.ORB_create(500)

img1_keys,disc1=orb.detectAndCompute(img1,None)
img2_keys,disc2=orb.detectAndCompute(img2,None)

matcher=cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches=matcher.match(disc1,disc2,None)
matches=list(matches)
matches.sort(key=lambda x:x.distance,reverse=False)
gdmt=int(len(matches)*0.15)
matches=matches[:gdmt]

img_matches=cv2.drawMatches(img1n,img1_keys,img2n,img2_keys,matches,None)

points1=np.zeros((len(matches),2),dtype=np.float32)
points2=np.zeros((len(matches),2),dtype=np.float32)

for i,match in enumerate(matches):
    points1[i,:]=img1_keys[match.queryIdx].pt
    points2[i, :] = img2_keys[match.trainIdx].pt

h,mask=cv2.findHomography(points2,points1,cv2.RANSAC)

img1_h,img1_w,chnl1=img1n.shape
img2_h,img2_w,chnl2=img2n.shape

img2_aligned=cv2.warpPerspective(img2n,h,(img2_w+img1_w,img2_h))
stiched=img2_aligned.copy()
stiched[0:img1_h,0:img1_w]=img1n

plt.imshow(stiched[:,:,::-1])

plt.show()
