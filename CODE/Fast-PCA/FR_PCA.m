clc
clear all;
load ORL_FaceDataSet;
A=double(ORL_FaceDataSet);

% Specifying no.of classes ... etc.
Num_Class=40; % No of classes
No_SampleClass=10; % No. of Samples/Class
No_TrainSamples=5; 
No_TestSamples=5;
DIM=10; % No. of PCs; can be changed from 1 to the total number of training samples
[TrainData, TestData]=Train_Test(A,No_SampleClass,No_TrainSamples,No_TestSamples);% Dataset separation: into training and testing sets
[m,n,TotalTrainSamples] = size(TrainData);
[m1,n1,TotalTestSamples] = size(TestData);
[TrainLabel,TestLabel]=LebelSamples(Num_Class, No_TrainSamples, No_TestSamples);% Labeling training & testing sets

%Training
TrainDataV = reshape(TrainData, [m*n TotalTrainSamples]);
MeanTrainDataV=(mean(TrainDataV'))'; % Here you can view mean image of the training set: MeanImageM=reshape(MeanTrainDataV,[m n]); imshow(MeanImageM,[]);
Diff=bsxfun(@minus,TrainDataV,MeanTrainDataV);% Centering training images
[EigVect1 EigVal]=eig_decomp(Diff'*Diff); % Computing and sorting EigenVectos 
EigVect=EigVect1(:,1:DIM);
EigImages=Diff*EigVect; % Here you can view the Eigenfaces as follows: for i=1 : DIM; EigenFace=reshape(EigImages(:,i),[m n]); imshow(EigenFace,[]); end
EigImages=bsxfun(@rdivide,EigImages,sqrt(sum(EigImages.^2))); % Normalizing EigenFaces 
TrainFeatureM=EigImages'*Diff; % Deriving training feature matrix

% Testing and Recognition 
TestResult = zeros(TotalTestSamples,1);
for i=1:TotalTestSamples
    TestImageC=reshape(TestData(:,:,i),m*n,1)- MeanTrainDataV;% Centering testing image
    TestFeatureV=EigImages'*TestImageC; % Deriving test feature vector
    DIST1=bsxfun(@minus,TrainFeatureM,TestFeatureV);% Computing differences between train feature matrix and test feature vector
    DIST=sqrt(sum(DIST1.*DIST1));% Computing distances
    [MINDIST ID]=min(DIST);% Returning min distance and its associated index
    % Note that: Lines 36 ~ 42,and 45 can Uncomment
    subplot 221; imshow(TestData(:,:,i),[]);title(['Tested Face = ' num2str(i)]);
    subplot 222; imshow(TrainData(:,:,ID),[]);title(['Recognized Face = ' num2str(ID)]);
    subplot(2,2,[3 4]); plot(DIST,'-o','MarkerIndices',[ID ID],'MarkerFaceColor','blue','MarkerSize',5);title(['Min Distance = ' num2str(MINDIST),' ID = ' num2str(ID)]);
    xlabel('Training Samples') 
    ylabel('Distance') 
    grid on
    grid minor
%     FF=getframe;
    TestResult(i) = TrainLabel(ID);
    pause % Press any key to continue or you can delay the process as you like; e.g., pause (0.2). 
end

Result = (TestResult == TestLabel);
CorrectRate = 100*sum(Result/(TotalTestSamples))




