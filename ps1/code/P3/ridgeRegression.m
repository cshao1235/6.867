function ridgeRegression()

    % Phi: matrix whose rows are data point inputs Phi(x)^(i)
    % y: vector whose entries are data points outputs y^(i)
    % lambda: parameter of ridge regression.  >0
    % return: value of w such that lambda * |w|^2 + sum_i |Phi(x)^(i)*w -
    % y^(i)|^2 is minimized
    function v = ridgeRegress(Phi, y, lambda)
        I = eye(length(Phi(1,:)));
        v = inv(lambda * I + Phi.' * Phi) * Phi.' * y;
    end

    
    function v = ridgeRegressPolynomialBasis(x, y, dimension, lambda)
        Phi = zeros([length(y) dimension]);
        for i=1:length(y)
            for j=1:dimension
                Phi(i,j)=x(i)^(j-1);
            end
        end
        v = ridgeRegress(Phi, y, lambda);
    end

    function part1test()
        X = [1;3;10];
        Y = [2;4;11];
        dimension = 2;
        lambda = 2;
        disp(ridgeRegressPolynomialBasis(X,Y,dimension,lambda));
    end

    function part1implementation()
        
        [X,Y]=loadFittingDataP2(0);
        dimension = 11;
        lambda = 1e-4;
        w = ridgeRegressPolynomialBasis(X.',Y.',dimension,lambda);
        disp(w);

        % plotting stuff
        hold on;
        plot(X.', Y.', 'o', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');

        x = 0 : 0.01 : 1;
        y = 0;
        for i = 1:dimension
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);
        hold off;
    end

    % call stuff here
    part1implementation();
end
