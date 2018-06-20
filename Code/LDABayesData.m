clc; clear all;
load('..\Data\data.mat')

%% initail variables
classNum = 200;
dimension = 504;
lowerDimensionto = classNum - 1;
numofTrainingSample = 400; 
numofTestingSample = 200;

fixtoInverseEigen = 250; %250~400
fixtoInverse = 0.75;
U = zeros(lowerDimensionto, classNum);
C = zeros(lowerDimensionto, lowerDimensionto, classNum);
Wi = zeros(lowerDimensionto, lowerDimensionto, classNum);
wi = zeros(lowerDimensionto, classNum);
wio = zeros(1,classNum);

%% Process LDA
% Get mi, m 
trainData = zeros(dimension, numofTrainingSample);
testData = zeros(dimension, numofTestingSample);
mis = zeros(dimension, classNum);
for i = 1 : classNum
    trainData(:, 2*i-1) = reshape(face(: , : , (3*i-2)),[dimension,1]);
    trainData(:, 2*i) = reshape(face(: , : , (3*i-1)),[dimension,1]);
    mis(:,i) = (trainData(:, 2*i-1) + trainData(:, 2*i))/2;
    testData(:, i) = reshape(face(: , : , (3*i)),[dimension,1]);
end

m = sum(trainData,2)/numofTrainingSample;

% Get SB
SB = 0;
for i = 1 : classNum
    Si = 2 * (mis(:, i) - m) * transpose(mis(:, i) - m);
    SB = SB + Si;
end

% Get SW
SW = 0;
for i = 1 : classNum
    Si = (trainData(:, 2*i-1) - mis(:, i)) * transpose(trainData(:, 2*i-1) - mis(:, i)) +...
         (trainData(:, 2*i) - mis(:, i)) * transpose(trainData(:, 2*i) - mis(:, i));
    SW = SW + Si;
end

% Find the corresponding eigenvector
%SB = SB + fixtoInverseEigen*eye(size(SB));
SW = SW + fixtoInverseEigen*eye(size(SW));
[W, LAMBDA] = eig(SB,SW);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda,'descend');
W = W(:,SortOrder);
eigenvectors = zeros(dimension, lowerDimensionto);
for j = 1: lowerDimensionto
        eigenvectors(:, j) = W(:, j);
end

% Get the data after transforamtion
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
