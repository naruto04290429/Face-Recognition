clc; clear all;
load('..\Data\data.mat')
%% Initailize variables
classNum = 200;
dimension = 504;
fixtoInverse = 0.75;
U = zeros(dimension, classNum);
C = zeros(dimension, dimension, classNum);
Wi = zeros(dimension, dimension, classNum);
wi = zeros(dimension, classNum);
wio = zeros(1,classNum);

%% Fist loop: Calaulate ui, convi using MLE
for i = 1 : classNum
    % Accorind to Maximun Likelihood Estimation
    train_data1 = reshape(face(: , : , (3*i-2)),[dimension,1]);
    train_data2 = reshape(face(: , : , (3*i-1)),[dimension,1]);
    
    U_i = ( train_data1 + train_data2 )/2;
    C_i = ((train_data1 - U_i ) * transpose( train_data1 - U_i ) +...
           (train_data2 - U_i ) * transpose( train_data2 - U_i ))/2;
    
    U(:, i) = U_i;
    C_i = C_i + fixtoInverse*eye(size(C_i));
    C(:, :, i) = C_i;
    % Quardratic Machine: Wi, wi, wip
    Wi(:, :, i) = (-1/2)*inv(C_i);
    wi(:, i) = inv(C_i)*U_i;
   
    wio(:, i) =  (-1/2)*( transpose(U_i)*inv(C_i)*U_i + log(det(C_i)) )+log(1/200); 
    
end

%% Second Loop: Run the gi(x) to classify

results = zeros(classNum,1);

for i = 1 : classNum
    max = intmin;
    X = reshape( face(:, :, 3*i) , [504,1]);
    for j = 1 : classNum
        G_j = transpose(X)*Wi(:, :, j)*X + transpose(wi(:, j))*X + wio(:, j);
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
