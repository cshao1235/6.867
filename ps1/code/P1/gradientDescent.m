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
    disp(111111111111111111111);
    disp(currentPoint);
    disp(norm(currentPoint));
    disp(grad(norm(currentPoint)));
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
function v = mymvnpdfgrad(mean, cov)
    function w = out(x)
        f = mymvnpdf(mean, cov);
        w = -f(x)*inv(cov)*(x - mean);
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

function part1implementation_gaussian()
[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
myfn = mymvnpdf(gaussMean, gaussCov);
myfngrad = mymvnpdfgrad(gaussMean, gaussCov);
startPoint = [1; 19];
stepSize = 1000000;
convergenceThreshold = 1.0e-12;
v = gradientDescent_1(myfn, myfngrad, startPoint, stepSize, convergenceThreshold);
disp(v);
end

function part2implementation()
[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
myfn = mymvnpdf(gaussMean, gaussCov);
myfngrad = mymvnpdfgrad(gaussMean, gaussCov);
myfnapproxgrad = approxGradient(myfn,0.1);
disp(myfngrad([40; 40]));
disp(myfnapproxgrad([40; 40]));
end

% call stuff here

end