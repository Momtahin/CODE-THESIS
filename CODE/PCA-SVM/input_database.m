% Function for reading the training database
function [image_vect ,class_vect, height, width]=input_database(img_path)
%Read the Image Databse folder
facedb=imageSet(img_path,'recursive');
%Initialise Variables
image_vect=[];
%Store image labels
class_vect=[];
%Height and width of images
height=0;
width=0;
channels=1;
classindex=1;

start=1;
subtract=2;
    
%Move through different images
for i=1:length(facedb)
    for j=start:(facedb(i).Count)-subtract
        img=read(facedb(i),j);
        [height, width, channels]=size(img);
        
        if(channels==3) %If the image is colored
            img = 0.2989*img(:,:,1)+0.5870*img(:,:,2)+0.1140*img(:,:,3) ;
        end
        
        %Reshape the image and store it in a vector
        img=reshape(img,1,height*width);
        image_vect=[image_vect;img];
        class_vect=[class_vect,classindex];
        
    end
     classindex=classindex+1;
end
classindex=classindex-1;
end
    