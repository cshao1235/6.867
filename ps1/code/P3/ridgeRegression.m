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

    
    function v = ridgeRegressPolynomialBasis(x, y, maxDegree, lambda)
        Phi = zeros([length(y) maxDegree]);
        for i=1:length(y)
            for j=1:maxDegree
                Phi(i,j)=x(i)^(j-1);
            end
        end
        v = ridgeRegress(Phi, y, lambda);
    end

    function part1test()
        X = [1;3;10];
        Y = [2;4;11];
        maxDegree = 2;
        lambda = 0;
        disp(ridgeRegressPolynomialBasis(X,Y,maxDegree,lambda));
    end

    function part1implementation()
        [X,Y]=loadFittingDataP2(1);
        disp(X.');
        disp(Y.');
        maxDegree = 3;
        lambda = 0;
        w = ridgeRegressPolynomialBasis(X.',Y.',maxDegree,lambda);
        disp(w);

        x = 0 : 0.01 : 1;
        y = w(1) * 1 + w(2) * x.^1 + w(3) * x.^2;
        plot(x,y)
        %TODO: plot curve with data points??
    end

    % call stuff here
    part1implementation();
end
