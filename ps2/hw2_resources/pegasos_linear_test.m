function pegasos_test

% load data from csv files
data = importdata(strcat('data/data3_train.csv'));
X = data(:,1:2);
Y = data(:,3);

disp('======Linear SVM======');
% Carry out training.

function train_linearSVM(X, Y, lambda, max_epochs)
    t = 0;
    w = zeros(size(X, 2), 1);
    w0 = 0.5;
    epoch = 0;
    while (epoch < max_epochs)
        for i = 1:size(X,1)
            t = t + 1;
            eta = 1/(t*lambda);
            if (Y(i)*(dot(w,X(i,:)) + w0) < 1)
                w = (1 - eta*lambda)*w + eta*Y(i)*(X(i,:)');
                w0 = w0 + eta*Y(i);
            else
                w = (1 - eta*lambda)*w;
            end
        end
        epoch = epoch + 1;
    end
end

epochs = 1000;
lambda = .02;
global w
global w0
disp(w);
disp(w0);
train_linearSVM(X, Y, lambda, epochs);


% Define the predict_linearSVM(x) function, which uses global trained parameters, w
function z = predict_linearSVM(x)
    z = dot(w, x) + w0;
end

hold on;
% plot training results
plotDecisionBoundary(X, Y, @predict_linearSVM, [-1,0,1], 'Linear SVM');
end



