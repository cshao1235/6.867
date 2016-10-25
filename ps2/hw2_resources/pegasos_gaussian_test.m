function pegasos_test

% load data from csv files
data = importdata(strcat('data/data3_train.csv'));
global X
X = data(:,1:2);
Y = data(:,3);
[n, d] = size(X);


disp('======Gaussian Kernel SVM======');
% Carry out training.
global alpha
global gamma
global K
epochs = 1000;
lambda = .02;
gamma = 2^2;

K = zeros(n,n);
%%% TODO: Compute the kernel matrix %%%
for i = 1:n
    for j = 1:n
        K(i, j) = exp(-1*gamma*(norm(X(i,:), X(j,:))*norm(X(i,:), X(j,:))));
    end
end
%%% TODO: Implement train_gaussianSVM %%%
function train_gaussianSVM(X, Y, lambda, epochs)
    t = 0;
    alphas = zeros(size(X, 2), 1);
    epoch = 0;
    while (epoch < epochs)
        for i = 1:size(X, 1)
            t = t + 1;
            eta = 1/(t*lambda)
            sum = 0;
            for j = 1:size(X, 1)
                sum = sum + alphas(j, 1)*K(j, i);
            end
            if(Y(i)*sum < 1)
                alphas(i) = (1 - eta*lambda)*alphas(i) + eta*Y(i);
            else
                alphas(i) = (1 - eta*lambda)*alphas(i)
            end
        end
        epoch = epoch + 1;
    end
end

train_gaussianSVM(X, Y, lambda, epochs);


% Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
function z = predict_gaussianSVM(x)
    z = dot(
end

hold on;

% plot training results
plotDecisionBoundary(X, Y, @predict_gaussianSVM, [-1,0,1], 'Gaussian Kernel SVM');


