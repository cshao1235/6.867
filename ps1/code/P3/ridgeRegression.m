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
        Phi = zeros([length(y) dimension+1]);
        for i=1:length(y)
            for j=1:dimension+1
                Phi(i,j)=x(i)^(j-1);
            end
        end
        v = ridgeRegress(Phi, y, lambda);
    end

    function v = squareError(X, Y, w)
        cost = 0;
        for i = 1:length(Y)
            phi_x = zeros([1 length(w)]);
            for j = 1:length(w)
                phi_x(j) = X(i)^(j-1);
            end
            cost = cost + (phi_x*w - Y(i))^2;
        end
        v = cost;
    end

    function part1implementation()
        [X,Y]=loadFittingDataP2(0);
        M = 10;
        lambda = 1;
        w = ridgeRegressPolynomialBasis(X.',Y.',M,lambda);
        disp(w);
        disp(squareError(X.',Y.',w));
        
        % plotting stuff
        hold on;
        plot(X.', Y.', 'o', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');

        x = 0 : 0.01 : 1;
        y = 0;
        for i = 1:M+1
            y = y + w(i) * x.^(i-1);
        end
        z = cos(pi*x) + cos(2*pi*x);
        plot(x,y);
        plot(x,z);
        hold off;
    end

    function part2implementation_ATrain()
        [AX,AY]=regressAData();
        [BX,BY]=regressBData();
        [VX,VY]=validateData();
        
        M = 9;
        lambda = .1;
        w = ridgeRegressPolynomialBasis(AX.',AY.',M,lambda);
        disp(w);
        disp(squareError(AX.',AY.',w));
        disp(squareError(BX.',BY.',w));
        %disp(squareError(VX.',VY.',w));
        
        % plotting stuff
        hold on;
        plot(AX.', AY.', 'o', 'MarkerSize', 8);
        plot(BX.', BY.', 'x', 'MarkerSize', 8);
        plot(VX.', VY.', '*', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');

        x = -3 : 0.01 : 2.5;
        y = 0;
        for i = 1:M
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);
        hold off;
    end

    function part2implementation_BTrain()
        [AX,AY]=regressAData();
        [BX,BY]=regressBData();
        [VX,VY]=validateData();
        
        M = 9;
        lambda = .1;
        w = ridgeRegressPolynomialBasis(BX.',BY.',M,lambda);
        disp(w);
        disp(squareError(AX.',AY.',w));
        disp(squareError(BX.',BY.',w));
        %disp(squareError(VX.',VY.',w));
        
        % plotting stuff
        hold on;
        plot(AX.', AY.', 'o', 'MarkerSize', 8);
        plot(BX.', BY.', 'x', 'MarkerSize', 8);
        plot(VX.', VY.', '*', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');

        x = -3 : 0.01 : 2.5;
        y = 0;
        for i = 1:M
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);
        hold off;
    end

    % call stuff here
    %part2implementation_ATrain();
    part1implementation();
end
