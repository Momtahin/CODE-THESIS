% Function for capturing live images and cropping the face
% Author: Shreyas Shubhankar
% Code assumes that only 1 face is found while detection

function[regi]=face_detect_live()
%Create a video input object
rgbvideo = videoinput('winvideo',1);
for j=1:1000
    %Get Snapshot
    gray=getsnapshot(rgbvideo);
    %Detect face using Viola-Jones detector algorithm
    faceDetector = vision.CascadeObjectDetector();
    %Get the bounding box of face
    bbox = faceDetector(gray);
    %Tag the face in the image
    fac = insertObjectAnnotation(gray,'rectangle',bbox(:,:),'Face');
    if isempty(bbox)==0   %If face is detected
        %Get the first and only bounding box
        i=1;
        %Get the co-ordinates and size of bounding box
        x=bbox(i,1);
        y=bbox(i,2);
        w=bbox(i,3);
        h=bbox(i,4);
        %Register the face
        regi=gray(y:y+w,x:x+h,:);
        regi=rgb2gray(regi);
        %Resize and crop to required size
        regi=imresize(regi,[112 112]);
        regi=regi(:,10:101);
        %Show the detected face in the original image
        imshow(fac)
        %Break if face has been recognised
        break;
    end
end
%figure; imshow(fac); title('Detected face');
end

