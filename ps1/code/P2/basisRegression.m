function basisRegression()

%returns the maximum likelihood weight vector for polynomial basis,
%assuming that observed values are perturbed by some Gaussian noise from
%such a polynomial basis
%X, list of data points for polynomial basis
%Y, values weights should be fitted to
%m, maximum order of polynomial basis
function v = maxLikelihoodPolynomialWeights(X, Y, m)
    s = '111111111111';
    n = size(Y, 2);
    x = zeros(n, m + 1);
    for r = 1:n
        for c = 1:(m + 1)
            x(r, c) = (X(r))^(c - 1);
        end
    end
    v = inv(x.'*x)*x.'*Y.';
end

function plotting()
    data = importdata('curvefittingp2.txt');

    X = data(1,:); Y = data(2,:);
    M = 10;
    
    function f = fittedPolynomial(X, Y, m, x)
        w = maxLikelihoodPolynomialWeights(X, Y, m);
        sum = 0;
        for i = 1:(m + 1)
            sum = sum + w(i)*(x^(i - 1));
        end
        f = sum;
    end
    
    x = (0:0.01:1);
    y = arrayfun(@(x) fittedPolynomial(X, Y, M, x), x);
%     disp(x);
%     f = fittedPolynomial(X, Y, 3, x);

    figure;
    plot(X, Y, 'o', 'MarkerSize', 8);
    hold on;
    plot(x,y);
%     plot(x, @(x) fittedPolynomial(X, Y, 3, x));
    hold off;
    xlabel('x'); ylabel('y');
end

plotting();
% [X, Y] = loadFittingDataP2();
% disp(maxLikelihoodPolynomialWeights(X, Y, 3));
    
end