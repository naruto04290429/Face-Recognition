clc; clear all;

%% initail variables
classNum = 68;
dimension = 48*40;

choice = 1;
if choice == 1
    load('..\Data\pose.mat'); %each image: 48*40, pose: 48x40x13x68
    totalSampleperClass = 13;
    trainingSampleperClass = 11;
    Data = reshape(pose, [dimension, totalSampleperClass, classNum]);
else
    load('..\Data\illumination.mat'); %each image: 48*40, pose: 48x40x21x68
    totalSampleperClass = 21;
    trainingSampleperClass = 16;
    Data = reshape(illum, [dimension, totalSampleperClass, classNum]);
end

testingSampleperClass = totalSampleperClass - trainingSampleperClass;
numofTrainingSample = trainingSampleperClass*classNum;
numofTestingSample = testingSampleperClass*classNum;

lowerDimensionto = 1000; 

fixtoInverse = 0.75;
U = zeros(lowerDimensionto, classNum);
C = zeros(lowerDimensionto, lowerDimensionto, classNum);
Wi = zeros(lowerDimensionto, lowerDimensionto, classNum);
wi = zeros(lowerDimensionto, classNum);
wio = zeros(1,classNum);

%% First loop: Parse data
trainData = zeros(dimension, trainingSampleperClass, classNum);
testData = zeros(dimension, testingSampleperClass, classNum);
for i = 1 : classNum
    for j = 1 : trainingSampleperClass
        trainData(:, j, i) = Data(:, j, i);
    end
    for j = 1 + trainingSampleperClass : totalSampleperClass
        testData(:, j-trainingSampleperClass, i) = Data(:, j, i);
    end
end
trainData = reshape(trainData, [dimension, numofTrainingSample]);
testData = reshape(testData, [dimension, numofTestingSample]);

%% Second loop: Process PCA
% Get mu of the data, and center the data
UData = sum(trainData,2)/numofTrainingSample;
for i = 1 : numofTrainingSample
    trainData(:, i) = trainData(:, i) - UData;
end
for i = 1 : numofTestingSample
    testData(:, i) = testData(:, i) - UData;
end

% Form C head, dimension * dimension matrix
C_h =  trainData * transpose(trainData) / numofTrainingSample;

% D: eigenvalue, V: eigenvector
[V,D] = eig(C_h);
D = eig(C_h);
[D_sort D_index] = sort(D,'descend');
V_sort=V(:, D_index);

% Get the corresponding eigenvector
eigenvectors = zeros(dimension, lowerDimensionto);
for j = 1 : lowerDimensionto
        eigenvectors(:, j) = V_sort(:, j);
end

% Get the traing data after transformation
trainData = transpose(eigenvectors) * trainData; 
testData = transpose(eigenvectors) * testData; 
 
%% Third loop: Calaulate ui, convi using MLE

for i = 1 : classNum
    % Accorind to Maximun Likelihood Estimation
    U_i = 0;
    C_i = 0;
    for j = 1 : trainingSampleperClass
        U_i = U_i + trainData(:,trainingSampleperClass*(i-1)+j);
        C_i = C_i + (trainData(:,trainingSampleperClass*(i-1)+j)- U_i) *...
                    transpose(trainData(:,trainingSampleperClass*(i-1)+j) - U_i);
    end
    U_i = U_i/trainingSampleperClass;
    C_i = C_i/trainingSampleperClass;
    
    C_i = C_i + fixtoInverse*eye(size(C_i));
    U(:, i) = U_i;
    C(:, :, i) = C_i;
    
    % Quardratic Machine: Wi, wi, wip
    Wi(:, :, i) = (-1/2)*inv(C_i);
    wi(:, i) = inv(C_i)*U_i;
    wio(:, i) = (-1/2)*( transpose(U_i)*inv(C_i)*U_i + log(det(C_i)) )+log(1/classNum); 
end
x = 0;

%% Fourth Loop: Run the gi(x) to classify

results = zeros(classNum,testingSampleperClass, 1);

for i = 1 : classNum
    max = intmin;
    for j = 1 : testingSampleperClass
        X = testData(:, testingSampleperClass*(i-1)+j);
        for k = 1 : classNum
            G_k = transpose(X)*Wi(:, :, k)*X + transpose(wi(:, k))*X + wio(:, k);
            if G_k > max
                max = G_k;
                results(i, j , 1) = k;
            end
        end
    end
end

correct = 0;
for i = 1 : classNum
    for j = 1 : testingSampleperClass
        if(i == results(i,j,1))
            correct = correct + 1;
        end
    end
end
disp(correct/numofTestingSample);