%% Facial Recognition Main Code
% Author: Shreyas Shubhankar
% Code optimised for AT&T Database
% Uses Eigenface (PCA) approach to get training features and SVM for
% training ML model
clc
clearvars
tic
facedb=imageSet('orl_faces','recursive');
disp('Reading the training database');
%Read all the images
[image_vect,class_vect,height,width]=input_database('orl_faces');
M=length(class_vect);
disp('Processing for PCA');
%Mean of each column and stored as row vector
mean_i=mean(image_vect,1);
%Convert image vector into double from uint8 and subtract 
%each image from mean
Xm = double(image_vect)-repmat (mean_i , size(image_vect,1),1) ;
disp('Calculating right singular vectors and singular values...');
% Calculate Right Singular Vectors and Singular Values
[U,S,V]=svd(Xm);
% Singular Values matrix will have at most M-1 non zero values
S=S(:,1:M-1);
% Choosing number of principal components to retain 99% variance
totalS=sum(diag(S));
varS=0;
for i= 1:M-1
    varS=varS+S(i,i);
    ratio=varS/totalS;
    if ratio>=0.99 %Change according to need
        disp(i);
        break;
    end
end
S=S(:,1:i);
V=V(:,1:i);
%Training Data
train=Xm*V;

toc
disp('Training/Loading Machine Learning Model...');
tic
%Use multi class SVM Classifier for making Machine Learning Model
mdl=fitcecoc(train,class_vect);
save('training_data.mat','mdl');   %Save and load ML model for future use
%load('training_data.mat');
toc
%% Calculate Accuracy
tic
accuracy=0;
disp('Calculating Accuracy');
for i=1:length(facedb)
   
    for j=9:10
        img=read(facedb(i),j);
        temp=img;
        img=reshape(temp,1,height*width);
        
        img=double(img);
        img=img-mean_i;
        projection=img*V;
        pre=predict(mdl,projection);
        if (pre==i)
            accuracy=accuracy+1;
        end
        temp2=read(facedb(pre),1);
        
         disp('Query and Matched images from Database');
         if j==9
             subplot(2,2,1);
             imshow(temp);
             title('Query');
             subplot(2,2,2);
             title('Matched');
             imshow(temp2);
             
         else
             subplot(2,2,3);
             imshow(temp);
             title('Query');
             subplot(2,2,4);
             title('Matched');
             imshow(temp2);
         end
            
    end
end

% Calculate accuracy
accuracy=100*(accuracy/(length(facedb)*2))
toc
%% Live Recognition of Face
tic
disp('Capturing Face');
%Capture Query image from camera
imgcam=face_detect_live();
%Store it temporarily
temp=imgcam;
%Reshape to row vector
imgcam=reshape(imgcam,1,height*width);
%Convert to double for manipulations
imgcam=double(imgcam);
%Calculate difference from mean
imgcam=imgcam-mean_i;
%Reprojection on Principal Component Vector Space
projection=double(imgcam)*V;
disp('Predicting a potential match');
pre=predict(mdl,projection);
%Show the query and matched faces
figure;
temp2=read(facedb(pre),1);
subplot(1,2,1);
imshow(temp);
title('Query');
subplot(1,2,2);
imshow(temp2);
title('Matched');
toc
disp('End of program');