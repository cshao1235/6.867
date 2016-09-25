function gradientDescent()

%Finds the x for which fn(x) is minimal by gradient descent
% fn: the (scalar) function to be minimized
% grad: a (vector) function, equal to the gradient of fn or an approximation
% startPoint: starting point for gradient descent
% step size: learning rate, so each step is stepSize * gradient(point)
% convergenceThreshold: determines when algorithm stops.  We terminate when
% |grad(point)| < convergenceThreshold
% return: point where gradient descent terminated
function w = gradientDescent_1(fn, grad, startPoint, stepSize, convergenceThreshold)
currentPoint = startPoint;
while (norm(grad(currentPoint)) > convergenceThreshold)
%    a = 'at point';
%    b = 'cost is';
%    c = 'gradient is';
%    d = 'gradient norm is';
%    disp(a);
%    disp(currentPoint.');
%    disp(b);
%    disp(fn(currentPoint).');
%    disp(c);
%    disp(grad(currentPoint).');
%    disp(d);
%    disp(norm(grad(currentPoint)));
    currentPoint = currentPoint - stepSize*grad(currentPoint);
end
w = currentPoint;
end

% outputs a multivariate gaussian distribution
% mean: mean vector
% cov: covariance matrix
% return: a function f such that f(x) is Gaussian(mean, cov)(x)
function v = mymvnpdf(mean, cov)
n = size(cov, 1);
    function w = out(x)
        w = -1/sqrt((2*pi)^n*det(cov))*exp(-1/2*(x - mean).'*inv(cov)*(x - mean));
    end
    v = @(x) out(x);
end

% outputs a multivariate gaussian distribution's gradient
% mean: mean vector of multivariate gaussian 
% cov: covariance matrix of multivariate gaussian
% return: a function f such that f(x) is gradient(Gaussian(mean, cov))(x)
function v = mymvnpdfGrad(mean, cov)
    function w = out(x)
        f = mymvnpdf(mean, cov);
        w = -f(x)*inv(cov)*(x - mean);
    end
    v = @(x) out(x);
end

% outputs a quadratic bowl
% A: matrix that determines quadratic term
% b: vector that determines linear term
% return: a function f such that f(x) is quadraticBowl(A, b)(x)
function v = myQuadBowl(A, b)
    function w = out(x)
        w = 1/2*x.'*A*x - x.'*b;
    end
v = @(x) out(x);
end

% outputs a quadratic bowl's gradient
% A: matrix that determines quadratic term
% b: vector that determines linear term
% return: a function f such that f(x) is gradient(quadraticBowl(A, b)(x))
function v = myQuadBowlGrad(A, b)
    function w = out(x)
        w = A*x - b;
    end
v = @(x) out(x);
end

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

% Cost function of batch gradient 
% X: matrix where columns are data points x^(i)
% y: vector whose entries are data points y^(i)
% return: a function f where f(theta) = sum |x^(i)*theta - y^(i)|^2
function v = batchCost(X, y)
    function w = out(theta)
        n = length(theta);
        cumulSum = 0;
        for k = 1:n
            x = X(k,:);
            cumulSum = cumulSum + (x * theta - y(k))^2;
            %disp(size(x * theta))
            %cumulSum = 0;
            %disp(x * theta);
        end
        w = cumulSum;
    end
    v = @(theta) out(theta);
end

function part1implementation_gaussian()
[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
myfn = mymvnpdf(gaussMean, gaussCov);
myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
startPoint = [1; 19];
stepSize = 1000000;
convergenceThreshold = 1.0e-12;
v = gradientDescent_1(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold);
disp(v);
end

function part1implementation_quadbowl()
[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
myfn = myQuadBowl(quadBowlA, quadBowlb);
myfnGrad = myQuadBowlGrad(quadBowlA, quadBowlb);
startPoint = [1; 30];
stepSize = 1.0e-3;
convergenceThreshold = 1.0e-4;
v = gradientDescent_1(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold);
disp(v);
end

function part2implementation()
[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
myfn = mymvnpdf(gaussMean, gaussCov);
myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
myfnApproxGrad = approxGradient(myfn,0.1);
disp(myfnGrad([40; 40]));
disp(myfnApproxGrad([40; 40]));
end

function part3implementation()
[X,y] = loadFittingDataP1();
epsilon = 1.0e-6;
batchErrorFn = batchCost(X,y);
batchErrorFnGrad = approxGradient(batchErrorFn, epsilon);
startPoint = zeros(size(X(1,:).')); % zero column matrix of size #columns of X
stepSize = 2e-5;
convergenceThreshold = 1e-6;
v = gradientDescent_1(batchErrorFn, batchErrorFnGrad, startPoint, stepSize, convergenceThreshold);
disp(v);
end


% call stuff here
part3implementation()

end