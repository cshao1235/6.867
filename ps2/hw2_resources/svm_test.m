function svm_test(name)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual
function sol = dualSVM(X, Y, C, kernel)
    n = size(Y, 1);
    H = zeros(n, n);
    for i = 1:n
        for j = 1:n
            H(i, j) = Y(i)*Y(j)*kernel(X(i,:),X(j,:));
        end
    end
    f = zeros(n, 1);
    for i = 1:n
        f(i, 1) = -1;
    end
    Aeq = zeros(1, n);
    for i = 1:n
        Aeq(i) = Y(i);
    end
    beq = zeros(1, 1);
    lb = zeros(n, 1);
    ub = zeros(n, 1);
    for i = 1:n
        ub(i, 1) = C;
    end
    optim_ver = ver('optim');
    optim_ver = str2double(optim_ver.Version);
    if optim_ver >= 6
        opts = optimset('Algorithm', 'interior-point-convex');
    else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
    end
    sol = quadprog(H, f, [], [], Aeq, beq, lb, [], [], opts);
end

function z = kernel(x, y)
    z = dot(x, y);
end

C = 1;
trainedParameters = dualSVM(X, Y, C, @(x,y) kernel(x, y));
d = size(X, 2);
n = size(X, 1);
w = zeros(d, 1);
for i = 1:n
    if(trainedParameters(i) ~= 0)
        w = w + trainedParameters(i)*Y(i)*X(i, :)';
    end
end
nonzeroCount = 0;
b = 0;
for i = 1:n
    if (trainedParameters(i) > 0)
        b = b + Y(i);
        for j = 1:n
            if (trainedParameters(j) > 0)
                b = b - trainedParameters(j)*Y(j)*kernel(X(i,:), X(j,:));
            end
        end
        nonzeroCount = nonzeroCount + 1;
    end
end
b = b/nonzeroCount;
disp(norm(w));

% Define the predictSVM(x) function, which uses trained parameters
function y = predictSVM()
    y = @(x) (dot(w, x) + b);
end


hold on;
% plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Train');


disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Validate');
hold off;
end

