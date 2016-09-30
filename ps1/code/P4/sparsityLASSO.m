function sparsityLASSO()
    
    function plotCurveGivenCoeffs(W)
        x = -1:0.01:1;
        y = W(1) * x;
        for i = 1:12
            y = y + W(i+1) * sin(0.4*pi*x*i);
        end
        plot(x,y);
    end

    % Phi: matrix whose rows are data point inputs Phi(x)^(i)
    % y: vector whose entries are data points outputs y^(i)
    % lambda: parameter of ridge regression.  >0
    % return: value of w such that lambda * |w|^2 + sum_i |Phi(x)^(i)*w -
    % y^(i)|^2 is minimized
    function v = ridgeRegress(Phi, y, lambda)
        I = eye(length(Phi(1,:)));
        v = inv(lambda * I + Phi.' * Phi) * Phi.' * y;
    end

    function part1Implementation()
        [x, y] = lassoTestData();
        n = length(y);
        M = 13;
        X = zeros([n M]);
        for r = 1:n
            X(r, 1) = x(r);
            for c = 2:M
                X(r, c) = sin(0.4*pi*x(r)*(c - 1));
            end
        end
        l = 1;
        lambda = zeros([l 1]);
        for r = 1:l
            lambda(r) = 10.0^(-r);
        end
        w = lasso(X, y, 'Lambda', lambda);
        %lassoPlot(w);
        disp(w);
        
        ww = size(w);
        cols = ww(2);

 
        %plotting
        xlabel('x'); ylabel('y');
        
        hold on;
        plot(x.',y.','o','MarkerSize',8);
        
        [xtest, ytest] = lassoTestData();
        plot(xtest.',ytest.','o','MarkerSize',8);
        
        [xval, yval] = lassoValData();
        plot(xval.',yval.','o','MarkerSize',8);

        W = lassoTrueData();
        plotCurveGivenCoeffs(W);

        for i = 1:cols
            coeffs = w(:,i);
            plotCurveGivenCoeffs(coeffs);
        end
        
        for r = 1:l
            ridge_w = ridgeRegress(X, y.', lambda(r));
            plotCurveGivenCoeffs(ridge_w);
        end
        
%        stupidFit = ridgeRegress(X, y.', 1e-14);
%        plotCurveGivenCoeffs(stupidFit);
        
       legend('Training', 'Test', 'Validation', 'True value','LASSO', 'Ridge Regression');
%         legend('Training', 'Test', 'Validation', 'True Curve', 'LASSO, lambda=0.001', 'LASSO, lambda=0.01', 'LASSO, lambda=0.1', 'LASSO, lambda=1')
%         hold off;
        ridge_w = ridgeRegress(X, y.', 100);
        %bar(ridge_w);
    end



    part1Implementation();
end