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
    count = 0;
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
        count = count+1;
        %    disp(count);
        %    disp(currentPoint);
        %    disp(grad(currentPoint));
        currentPoint = currentPoint - stepSize*grad(currentPoint);
    end
    disp(count);
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

% Cost function of batch gradient 
% X: vector = a data point x^(i)
% y: scalar = a data point y^(i)
% return: a function f where f(theta) = |x^(i)*theta - y^(i)|^2
function v = singlePointCost(x, y)
    function w = out(theta)
        w = (x * theta - y)^2;
    end
    v = @(theta) out(theta);
end

function part1implementation_gaussian()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = mymvnpdf(gaussMean, gaussCov);
    myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
    startPoint = [0;0];
    stepSize = 1e6;
    convergenceThreshold = 1e-11;
    v = gradientDescent_1(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold);
    disp(norm(v-[10;10]));
end

function part1implementation_quadbowl()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = myQuadBowl(quadBowlA, quadBowlb);
    myfnGrad = myQuadBowlGrad(quadBowlA, quadBowlb);
    startPoint = [0; 0];
    stepSize = 1e-2;
    convergenceThreshold = 1e-3;
    v = gradientDescent_1(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold);
    disp(norm(v-[80/3;80/3]));
end

function part2implementation()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = mymvnpdf(gaussMean, gaussCov);
    myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
    myfnApproxGrad = approxGradient(myfn,0.1);
    disp(myfnGrad([40; 40]));
    disp(myfnApproxGrad([40; 40]));
end

function part3implementation_batch()
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

function part3implementation_stochastic()
    [X,y] = loadFittingDataP1();
    n = length(y);
    epsilon = 1.0e-6;
    startTheta = zeros(size(X(1,:).')); % zero column matrix of size #columns of X
    convergenceThreshold = 1;
    objectiveCost = batchCost(X, y);
    objectiveGradient = approxGradient(objectiveCost,epsilon);
    
    theta = startTheta;
    prevTheta = startTheta;
    t = 0;
    tau = 0;
    kappa = .55;
    
    while (t== 0 || abs(objectiveCost(prevTheta) - objectiveCost(theta)) > convergenceThreshold)
        t = t+1;
        learningRate = 3e-5*(tau+t)^(-kappa);
        for k = 1:n
%            e = 'time is';
%            a = 'at point';
%            b = 'cost is';
%            c = 'gradient is';
%            d = 'gradient norm is';
%            f = 'stochastic gradient norm is';
%            g = 'learning rate is';
%            disp(e);
%            disp((t-1)*n + k);
%            disp(a);
%            disp(theta.');
%            disp(b);
%            disp(objectiveCost(theta).');
%            disp(c);
%            disp(objectiveGradient(theta).');
%            disp(d);
%            disp(norm(objectiveGradient(theta)));

            xi = X(k, :);
            yi = y(k);
            cost = singlePointCost(xi,yi);    
            gradCost =  approxGradient(cost, epsilon);

%            disp(f);
%            disp(norm(gradCost(theta)));
%            disp(g);
%            disp(learningRate);
            
            prevTheta = theta;
            theta = theta - learningRate * gradCost(theta);
        end
    end
    disp(theta);
end





%Finds the x for which fn(x) is minimal by gradient descent
% fn: the (scalar) function to be minimized
% grad: a (vector) function, equal to the gradient of fn or an approximation
% startPoint: starting point for gradient descent
% step size: learning rate, so each step is stepSize * gradient(point)
% convergenceThreshold: determines when algorithm stops.  We terminate when
% |grad(point)| < convergenceThreshold
% return: (point where gradient descent terminated, # iterations)
function w = gradientDescent_2(fn, grad, startPoint, stepSize, convergenceThreshold,actualAnswer)
    currentPoint = startPoint;
    count = 0;
    while (norm(grad(currentPoint)) > convergenceThreshold)
        count = count+1;
        if (count==5000)
            break;
        end
        currentPoint = currentPoint - stepSize*grad(currentPoint);
    end
    if norm(currentPoint-actualAnswer) > 1e20
        w = [NaN; count];
    else
        w = [norm(currentPoint-actualAnswer); count];
    end
end

function part1implementation_gaussian_x()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = mymvnpdf(gaussMean, gaussCov);
    myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
    startPoint = [0;0];
    stepSize = 1e6;
    convergenceThreshold = 1e-9;
    %validlambda = [3e5;1e6;1e6;1e7;3e7;1e8];
    
    predists = zeros([101 1]);
    dists = zeros([101 1]);
    iterations = zeros([101 1]);
    
    for i = 0:100    
        disp(i);
        
        predist = 10^(0.03*i);
        predists(i+1)=predist;
        
        startPoint = [10+predist/2^0.5; 10+predist/2^0.5];
        
        v = gradientDescent_2(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold, [10;10]);
        disp(v);
    
        iterations(i+1) = v(2);
        disp(v(2));
        
        dists(i+1) = v(1);
        disp(v(1));
%        disp(norm(v(1)-[10;10])); 
        
    end
    
    disp(predists);
    disp(iterations);
    disp(dists);

    title('Mutivariate Gaussian - varying start point');
    set(gca,'xscale','log');
    xlabel('distance from start point to goal'); 
    
    yyaxis left;
    loglog(predists, iterations, 'o', 'MarkerSize', 5);
    ylabel('iterations');
    
    yyaxis right;
    loglog(predists, dists, 'x', 'MarkerSize', 5);
    ylabel('accuracy');

end

function part1implementation_gaussian_lambda()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = mymvnpdf(gaussMean, gaussCov);
    myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
    startPoint = [0;0];
    stepSize = 1e6;
    convergenceThreshold = 1e-9;
    %validlambda = [3e5;1e6;1e6;1e7;3e7;1e8];
    
    lambdas = zeros([101 1]);
    dists = zeros([101 1]);
    iterations = zeros([101 1]);
    
    for i = 0:100    
        disp(i);
        
        stepSize = 10^(5+0.03*i);
        lambdas(i+1)=stepSize;
        
        v = gradientDescent_2(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold, [10;10]);
        disp(v);
    
        iterations(i+1) = v(2);
        disp(v(2));
        
        dists(i+1) = v(1);
        disp(v(1));
%        disp(norm(v(1)-[10;10])); 
        
    end
    
    disp(lambdas);
    disp(iterations);
    disp(dists);

    title('Mutivariate Gaussian - varying step size');
    set(gca,'xscale','log');
    xlabel('step size'); 
    
    yyaxis left;
    loglog(lambdas, iterations, 'o', 'MarkerSize', 5);
    ylabel('iterations');
    
    yyaxis right;
    loglog(lambdas, dists, 'x', 'MarkerSize', 5);
    ylabel('accuracy');

end


function part1implementation_gaussian_epsilon()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = mymvnpdf(gaussMean, gaussCov);
    myfnGrad = mymvnpdfGrad(gaussMean, gaussCov);
    startPoint = [0;0];
    stepSize = 1e6;
    convergenceThreshold = 1e-9;
    %validlambda = [3e5;1e6;1e6;1e7;3e7;1e8];
    
    epsilons = zeros([101 1]);
    dists = zeros([101 1]);
    iterations = zeros([101 1]);
    
    for i = 0:100    
        disp(i);
        
        convergenceThreshold = 10^(-12+0.06*i);
        epsilons(i+1)=convergenceThreshold;
        
        v = gradientDescent_2(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold,[10;10]);
        disp(v);
    
        iterations(i+1) = v(2);
        disp(v(2));
        
        dists(i+1) = v(1);
        disp(v(1));
%        disp(norm(v(1)-[10;10])); 
        
    end
    
    disp(epsilons);
    disp(iterations);
    disp(dists);

    title('Mutivariate Gaussian - varying convergence threshold');
    set(gca,'xscale','log');
    xlabel('convergence threshold'); 
    
    yyaxis left;
    loglog(epsilons, iterations, 'o', 'MarkerSize', 5);
    ylabel('iterations');
    
    yyaxis right;
    loglog(epsilons, dists, 'x', 'MarkerSize', 5);
    ylabel('accuracy');

end

function part1implementation_quadbowl_x()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = myQuadBowl(quadBowlA, quadBowlb);
    myfnGrad = myQuadBowlGrad(quadBowlA, quadBowlb);
    startPoint = [0; 0];
    stepSize = 1e-2;
    convergenceThreshold = 1;
   
    predists = zeros([101 1]);
    dists = zeros([101 1]);
    iterations = zeros([101 1]);
    
    for i = 0:100    
        disp(i);
        
        predist = 10^(0.05*i);
        predists(i+1)=predist;
        
        startPoint = [10+predist/2^0.5; 10+predist/2^0.5];
        
        v = gradientDescent_2(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold, [80/3;80/3]);
        disp(v);
    
        iterations(i+1) = v(2);
        disp(v(2));
        
        dists(i+1) = v(1);
        disp(v(1));
%        disp(norm(v(1)-[10;10])); 
        
    end
    
    disp(predists);
    disp(iterations);
    disp(dists);

    title('Quadratic Bowl - varying start point');
    set(gca,'xscale','log');
    xlabel('distance from start point to goal'); 
    
    yyaxis left;
    loglog(predists, iterations, 'o', 'MarkerSize', 5);
    ylabel('iterations');
    
    yyaxis right;
    loglog(predists, dists, 'x', 'MarkerSize', 5);
    ylabel('accuracy');

end

function part1implementation_quadbowl_lambda()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = myQuadBowl(quadBowlA, quadBowlb);
    myfnGrad = myQuadBowlGrad(quadBowlA, quadBowlb);
    startPoint = [0; 0];
    stepSize = 1e-2;
    convergenceThreshold = 1;
   
    lambdas = zeros([101 1]);
    dists = zeros([101 1]);
    iterations = zeros([101 1]);
    
    for i = 0:100    
        disp(i);
        
        stepSize = 10^(-3+0.03*i);
        lambdas(i+1)=stepSize;
        
        v = gradientDescent_2(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold,[80/3;80/3]);
        disp(v);
    
        iterations(i+1) = v(2);
        disp(v(2));
        
        dists(i+1) = v(1);
        disp(v(1));
%        disp(norm(v(1)-[10;10])); 
        
    end
    
    disp(lambdas);
    disp(iterations);
    disp(dists);

    title('Quadratic Bowl - varying step size');
    set(gca,'xscale','log');
    xlabel('step size'); 
    
    yyaxis left;
    loglog(lambdas, iterations, 'o', 'MarkerSize', 5);
    ylabel('iterations');
    
    yyaxis right;
    loglog(lambdas, dists, 'x', 'MarkerSize', 5);
    ylabel('accuracy');

end

function part1implementation_quadbowl_epsilon()
    [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
    myfn = myQuadBowl(quadBowlA, quadBowlb);
    myfnGrad = myQuadBowlGrad(quadBowlA, quadBowlb);
    startPoint = [0; 0];
    stepSize = 1e-2;
    convergenceThreshold = 1;
   
    epsilons = zeros([101 1]);
    dists = zeros([101 1]);
    iterations = zeros([101 1]);
    
    for i = 0:100    
        disp(i);
        
        convergenceThreshold = 10^(-4+0.06*i);
        epsilons(i+1)=convergenceThreshold;
        
        v = gradientDescent_2(myfn, myfnGrad, startPoint, stepSize, convergenceThreshold,[80/3;80/3]);
        disp(v);
    
        iterations(i+1) = v(2);
        disp(v(2));
        
        dists(i+1) = v(1);
        disp(v(1));
%        disp(norm(v(1)-[10;10])); 
        
    end
    
    disp(epsilons);
    disp(iterations);
    disp(dists);

    title('Quadratic Bowl - varying convergence threshold');
    set(gca,'xscale','log');
    xlabel('convergence threshold'); 
    
    yyaxis left;
    loglog(epsilons, iterations, 'o', 'MarkerSize', 5);
    ylabel('iterations');
    
    yyaxis right;
    loglog(epsilons, dists, 'x', 'MarkerSize', 5);
    ylabel('accuracy');

end





% call stuff here
%part3implementation_batch()
%part3implementation_stochastic()

part1implementation_quadbowl_x();

end