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
        
        validM = [3; 7; 11];
        validLambda = [1e-3; 1e-2; 1e-1; 1;1.3];
        
        M = 3;
        lambda = 1e-1;
        for j=1:5
            for i=1:3
                M = validM(i);
                lambda = validLambda(j);
                w = ridgeRegressPolynomialBasis(AX.',AY.',M,lambda);
                %disp(w);
                disp(M-1);
                disp(lambda);
                disp(squareError(AX.',AY.',w));
                disp(squareError(VX.',VY.',w));
                disp(squareError(BX.',BY.',w));
            end
        end
%        plotting stuff
        hold on;
        plot(AX.', AY.', 'o', 'MarkerSize', 8);
        plot(VX.', VY.', 'o', 'MarkerSize', 8);
        plot(BX.', BY.', 'o', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');
                
        w = ridgeRegressPolynomialBasis(AX.',AY.',3,1e-3).';
        xx = -3 : 0.01 : 2.5;
        yy = 0;
        for i = 1:4
            yy = yy+ w(i)*xx.^(i-1);
        end
        plot(xx,yy);
        
        w = ridgeRegressPolynomialBasis(AX.',AY.',7,1e-1).';
        xx = -3 : 0.01 : 2.5;
        yy = 0;
        for i = 1:8
            yy = yy+ w(i)*xx.^(i-1);
        end
        plot(xx,yy);
        
        w = ridgeRegressPolynomialBasis(AX.',AY.',11,1).';
        xx = -3 : 0.01 : 2.5;
        yy = 0;
        for i = 1:12
            yy = yy+ w(i)*xx.^(i-1);
        end
        plot(xx,yy);
        legend('Training','Validation','Test','M=3, lambda = 0.001','M=7, lambda=0.1','M=11, lambda=1');
        hold off;
    end

    function part2implementation_BTrain()
        [AX,AY]=regressAData();
        [BX,BY]=regressBData();
        [VX,VY]=validateData();
                
        validM = [3; 7; 11];
        validLambda = [1e-3; 1e-2; 1e-1; 1];
        
        for j=1:4
            for i=1:3
        
                M = validM(i);
                lambda = validLambda(j);
                w = ridgeRegressPolynomialBasis(BX.',BY.',M,lambda);
                %disp(w);
                disp(M-1);
                disp(lambda);
                disp(squareError(BX.',BY.',w));
                disp(squareError(VX.',VY.',w));
                disp(squareError(AX.',AY.',w));
            end
        end
        % plotting stuff
        hold on;
        plot(BX.', BY.', 'o', 'MarkerSize', 8);
        plot(VX.', VY.', 'o', 'MarkerSize', 8);
        plot(AX.', AY.', 'o', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');

        w = ridgeRegressPolynomialBasis(BX.',BY.',3,1);
        x = -3 : 0.01 : 2.5;
        y = 0;
        for i = 1:4
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);

        w = ridgeRegressPolynomialBasis(BX.',BY.',7,1e-3);
        x = -3 : 0.01 : 2.5;
        y = 0;
        for i = 1:8
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);

        w = ridgeRegressPolynomialBasis(BX.',BY.',11,1e-1);
        ylim([-3 4]);
        x = -3 : 0.01 : 2.5;
        y = 0;
        for i = 1:12
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);
        legend('Training','Validation','Test','M=3, lambda = 1','M=7, lambda=0.001','M=11, lambda=0.1');

        hold off;
    end

    % call stuff here
    part2implementation_BTrain();
    %part2implementation_BTrain();
end
