clc;
clear all;
warning off;
chos=0;
possibility=7;
while chos~=possibility,
    chos=menu('AUTOMATED MALARIA PARASITE DETECTION ','LOAD IMAGE','PREPROCESS','EXTRACTION','SEGMENTATION','MORPHOLOGY OPERATION','TOTAL CELLS','TOTAL INFECTED','break','show planes');

if chos==1,
[f,p]=uigetfile();
I=imread([p,f]);
figure,imshow(I),title('input image');
end


if chos==2,
%% RGB2gray image 
I2=rgb2gray(I);
subplot(3,3,1);imshow(I2);title('RGB TO GRAYSCALE');
%% Preprocessing 


I3=medfilt2(I2,[3 3]);
subplot(3,3,2);imshow(I3);title('FILTERED IMAGE');

y1=histeq(I3);
subplot(3,3,3);imshow(y1);title('HISTOGRAMED IMAGE');

end
if chos==9,
    subplot(2,2,1);
    imshow(I(:,:,3))
    title('blue');
    subplot(2,2,2);
    imshow(0.5*(I(:,:,2)))
    title('green');
    subplot(2,2,3);
    imshow(0.5*(I(:,:,1)))
    title('red');
    temp=(0.5*(I(:,:,1)));
    disp(temp(size(temp,1),size(temp,2)))
    temp1=(0.5*(I(:,:,2)));
    disp(temp1(size(temp1,1),size(temp1,2)))
    temp2=(I(:,:,3));
    disp(temp2(size(temp2,1),size(temp2,2)))
end
    
   
if chos==3,
%% Extracting the blue plane 
bPlane = I(:,:,3)  - 0.5*(I(:,:,1)) - 0.5*(I(:,:,2));
disp(bPlane(size(bPlane,1),size(bPlane,2)))
subplot(2,2,1);
imshow(bPlane);title('EXTRACTION-1');

%% Extract out purple cells

BW = bPlane > 29;
subplot(2,2,2);
imshow(BW);
title('EXTRACTION-2');
end
if chos==4,
%% Remove noise 100 pixels or less
BW2 = bwareaopen(BW, 100);
subplot(2,2,1)
imshow(BW2);
title('NOISE REMOVAL');

%% contrast 
I5=imadjust(I3);
subplot(2,2,2);imshow(I5);
title('INTENSITY ADJUSTMENT');
f=graythresh(I2)

I6=im2bw(I5,f);
subplot(2,2,3);imshow(I6);
title('BINARY IMAGE');

end
if chos==5,
%%morphology
I7=bwareaopen(I6,50)
subplot(3,3,1);imshow(I7)
title('AREA OPENING');
 [~, threshold] = edge(I2, 'sobel');
fudgeFactor = .5;
BWs = edge(I2,'sobel', threshold * fudgeFactor);
subplot(3,3,2);
imshow(BWs), title('GRADIENT MASK');
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);

BWsdil = imdilate(BWs, [se90 se0]);
subplot(3,3,3);
imshow(BWsdil);
title('DILATION');
BWdfill = imfill(BWsdil, 'holes');
subplot(3,3,4);imshow(BWdfill);
title('HOLES FILLING')


I2=rgb2gray(I);
I_eq = adapthisteq(I2);
subplot(3,3,5);imshow(I_eq);
title('ADAPTIVE HISTOGRAM');
bw = im2bw(I_eq, graythresh(I_eq));
 %%figure, imshow(bw)
 bw2 = imfill(bw,'holes');
bw3 = imopen(bw2, ones(5,5));
bw4 = bwareaopen(bw3, 40);
bw4_perim = bwperim(bw4);
overlay1 = imoverlay(I_eq, bw4_perim,[.3 1 .3]);
subplot(3,3,6);imshow(overlay1);
title('overlay of original image');
mask_em = imextendedmax(I_eq, 30);
subplot(3,3,9);
imshow(mask_em);
title('initial maxima transform')
%%title('MASKED IMAGE');
mask_em = imclose(mask_em, ones(5,5));
mask_em = imfill(mask_em, 'holes');
mask_em = bwareaopen(mask_em, 40);
subplot(3,3,7);
imshow(mask_em);title('masked image');
overlay2 = imoverlay(I_eq, bw4_perim | mask_em,[.3 1 .3]);
subplot(3,3,8);imshow(overlay2);
title('overlay of masked image');
end
if chos==6,


I9=bwlabel(mask_em);
RBCCOUNT=max(max(I9))
figure,imshow(I9)
title('TOTAL NO OF CELLS')

I8=imfill(I7,'holes');
figure,imshow(I8)
end
if chos==7,

L=bwlabel(I8);
%% Superimpose onto original image
figure, imshow(I), hold on
himage = imshow(I8);
set(himage, 'AlphaData', 0.5);
totalinfectedcells=max(max(L))

title('TOTAL NO OF INFECTED CELLS')

end

if chos==8,
    break
    
end

end