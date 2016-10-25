function svm_test(name)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual
function sol = dualSVM(X, Y, C)
    n = size(Y, 2);
    H = zeros(n, n);
    for i = 1:n
        for j = 1:n
            H(i, j) = Y(i)*Y(j)*dot(X(i),X(j));
        end
    end
    f = zeros(1, n);
    for i = 1:n
        f(i) = -1;
    end
    A = zeros(1, n);
    b = zeros(1, 1);
    Aeq = zeros(1, n);
    for i = 1:n
        Aeq(i) = Y(i);
    end
    beq = zeros(1, 1);
    lb = zeros(n, 1);
    ub = zeros(n, 1);
    for i = 1:n
        ub(i) = C;
    end
%     optim_ver = ver('optim');
%     disp(optim_ver);
%     optim_ver = str2double(optim_ver.Version);
%     if optim_ver >= 6
%         opts = optimset('Algorithm', 'interior-point-convex');
%     else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
%     end
    sol = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], opts);
end

trainedParameters = dualSVM(X, Y, 10000);

% Define the predictSVM(x) function, which uses trained parameters
function y = predictSVM()
    d = size(X, 1);
    n = size(X, 2);
    w = zeros(d);
    for i = 1:n
        w = w + trainedParameters(i)*X(i);
    end
    for i = 1:n
        if (trainedParameters(i) < C && trainedParameters(i) > 0)
            b = 1.0/Y(i) - dot(w, X(i));
        end
    end
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

