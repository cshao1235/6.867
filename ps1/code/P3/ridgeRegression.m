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
        x = [1;3;10];
        y = [2;4;11];
        maxDegree = 2;
        lambda = 0;
        disp(ridgeRegressPolynomialBasis(x,y,maxDegree,lambda));
    end

    % call stuff here
    part1test();
end
