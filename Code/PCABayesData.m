clc; clear all;
load('..\Data\data.mat')

%% initail variables
classNum = 200;
dimension = 504;
lowerDimensionto = 200; 
numofTrainingSample = 400;
numofTestingSample = 200;

fixtoInverse = 0.75;
U = zeros(lowerDimensionto, classNum);
C = zeros(lowerDimensionto, lowerDimensionto, classNum);
Wi = zeros(lowerDimensionto, lowerDimensionto, classNum);
wi = zeros(lowerDimensionto, classNum);
wio = zeros(1,classNum);

%% First loop: Process PCA
trainData = zeros(dimension, numofTrainingSample);
testData = zeros(dimension, numofTestingSample);
% Get mu of the data, and center the data
for i = 1 : classNum
    trainData(:, 2*i-1) = reshape(face(: , : , (3*i-2)),[dimension,1]);
    trainData(:, 2*i) = reshape(face(: , : , (3*i-1)),[dimension,1]);
    testData(:, i) = reshape(face(: , : , (3*i)),[dimension,1]);
end

UData = sum(trainData,2)/numofTrainingSample;

for i = 1 : numofTrainingSample
    trainData(:, i) = trainData(:, i) - UData;
end

for i = 1 : numofTestingSample
    testData(:, i) = testData(:, i) - UData;
end

% Form C head, dimension * dimension matrix
C_h =  trainData * transpose(trainData) /numofTrainingSample;

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

%% Second loop: Calaulate ui, convi using MLE
for i = 1 : classNum
    % Accorind to Maximun Likelihood Estimation
    train_data1 = trainData(:, 2*i-1);
    train_data2 = trainData(:, 2*i);
    
    U_i = ( train_data1 + train_data2 )/2;
    C_i = ((train_data1 - U_i ) * transpose( train_data1 - U_i ) +...
           (train_data2 - U_i ) * transpose( train_data2 - U_i ))/2;
    
    U(:, i) = U_i;
    C_i = C_i + fixtoInverse*eye(size(C_i));
    C(:, :, i) = C_i;
    % Quardratic Machine: Wi, wi, wip
    Wi(:, :, i) = (-1/2)*inv(C_i);
    wi(:, i) = inv(C_i)*U_i;
   
    wio(:, i) =  (-1/2)*( transpose(U_i)*inv(C_i)*U_i + log(det(C_i)) )+log(1/classNum); 
end

%% Third loop: Run the gi(x) to classify

results = zeros(classNum,1);

for i = 1 : classNum
    max = intmin;
    X = testData(:, i);
    
    for j = 1 : classNum
        G_j = transpose(X)*Wi(:, :, j)*X + transpose(wi(:, j))*X + wio(:, j);      
        %disp(G_j);
        if G_j > max
            max = G_j;
            results(i,1) = j;
        end
    end
end

correct = 0;
for i = 1 : classNum
    if(i == results(i,1))
        correct = correct + 1;
    end
end
disp(correct/classNum);
