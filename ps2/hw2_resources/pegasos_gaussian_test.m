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
gamma = 2^1;

function z = gaussianKernel(x1, x2)
    z = exp(-1*(norm(x1 - x2)*norm(x1 - x2))/gamma);
end

K = zeros(n,n);
%%% TODO: Compute the kernel matrix %%%
for i = 1:n
    for j = 1:n
        K(i, j) = gaussianKernel(X(i,:), X(j,:));
    end
end
%%% TODO: Implement train_gaussianSVM %%%
function train_gaussianSVM(X, Y, lambda, epochs)
    t = 0;
    alpha = zeros(size(X, 1), 1);
    epoch = 0;
    while (epoch < epochs)
        for i = 1:size(X, 1)
            t = t + 1;
            eta = 1/(t*lambda);
            sum = 0;
            for j = 1:size(X, 1)
                sum = sum + alpha(j, 1)*K(j, i);
            end
            if(Y(i)*sum < 1)
                alpha(i) = (1 - eta*lambda)*alpha(i) + eta*Y(i);
            else
                alpha(i) = (1 - eta*lambda)*alpha(i);
            end
        end
        epoch = epoch + 1;
    end
end

train_gaussianSVM(X, Y, lambda, epochs);


% Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
function z = predict_gaussianSVM(x)
    sum = 0;
    for i = 1:size(X, 1)
        sum = sum + alpha(i)*gaussianKernel(X(i,:), x);
    end
    z = sum;
end

hold on;

% plot training results
plotDecisionBoundary(X, Y, @predict_gaussianSVM, [-1,0,1], 'Gaussian Kernel SVM');
hold off;
end


