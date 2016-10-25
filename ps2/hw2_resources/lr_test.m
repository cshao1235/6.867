function lr_test(name)
disp('======Training======');
% load data from csv files

% data = importdata('data1_train.csv');
data = importdata(strcat('data/data',name,'_train.csv'));

X = data(:,1:2);
Y = data(:,3);

% Carry out training.

% outputs an approximation of a function's gradient
% fn: function whose gradient is to be computed
% epsilon: half-width of finite difference used to compute gradient
% return: a function f such that f(x) approximates gradient(fn)(x)
function v = approxGradient(fn, epsilon)
    function w = out(x)
        n = length(x);
        grad = zeros(size(x));
        for k = 1:n
            smallvector = zeros(size(x));
            smallvector(k) = smallvector(k) + epsilon;
            grad(k) = (fn(x+smallvector)-fn(x-smallvector)) / (2*epsilon);
        end
        w = grad;
    end
    v = @(x) out(x);
end

function y = gradientDescent(fn, grad, startPoint, stepSize, convergenceThreshold, maxIterations)
    currentPoint = startPoint;
    prevPoint = currentPoint;
    count = 0;
    X1 = zeros(maxIterations, 1);
    Y1 = zeros(maxIterations, 1);
    while (count == 0 || abs(fn(prevPoint) - fn(currentPoint)) > convergenceThreshold)
        count = count+1;
%         disp(count);
        disp(currentPoint);
        if (count==maxIterations)
            break;
        end
        prevPoint=currentPoint;
        g = grad(currentPoint);
        X1(count)=count;
        Y1(count)=norm(currentPoint(1:(length(currentPoint) - 1)));
%         disp(g);
        currentPoint = currentPoint - stepSize*g;
    end
    hold on;
    title('Norm of weight vector');
    scatter(X1(2:count), Y1(2:count));
    hold off;
    y = currentPoint;
end

function y = nll(X, Y, w)
    d = size(X,2);
    n = size(Y,1);
    sum = 0;
    for i = 1:n
        X1 = zeros(d, 1);
        for j = 1:d
            X1(j) = X(i,j);
        end
        X1(d + 1) = 1;
        innerDot = dot(w, X1);
        sum = sum + log(1 + exp(-1.0*Y(i)*innerDot));
    end
    y = sum;
end

function y = error(X, Y, lambda)
    function w = error1(w1)
        n = length(w1) - 1;
        z = zeros(n, 1);
        for i = 1:n;
           z(i) = w1(i);
        end
        nm = norm(z, 2);
        w = nll(X, Y, w1) + lambda*nm*nm;
    end
    y = @(w1) error1(w1);
end

% part1implementation();

myfn = error(X, Y, 1);
myfnGrad = approxGradient(myfn, 1e-6);
startPoint = [0.1; 0.1; 0.1];
stepSize = 1e-2;
convergenceThreshold = 1e-7;
maxIterations = 500;
classifier = gradientDescent(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold, maxIterations);

disp(classifier);

% Define the predictLR(x) function, which uses trained parameters
function y = predictLR()
%     y = dot(x, classifier(1:2)) + classifier(3);
    function z = score(x)
        w = [classifier(1) classifier(2)];
        v = dot(x, w) + classifier(3);
        z = 1.0/(1.0 + exp(-1.0*v));
    end
    y = @(x) score(x);
end

hold on;

% plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5 0.5], 'LR Train');

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5 0.5], 'LR Validate');
hold off;
end
