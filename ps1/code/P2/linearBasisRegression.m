function linearBasisRegression()

    %%%%%%%%%%%%%%%%%%%%
    %   GENERAL STUFF  %
    %%%%%%%%%%%%%%%%%%%%
    
    % Phi: matrix whose rows are data point inputs Phi(x)^(i)            
    % y: column vector whose entries are data points outputs y^(i)       
    % return: value of w such that lambda * |w|^2 + sum_i |Phi(x)^(i)*w- 
    % y^(i)|^2 is minimized                                              
    function v = linRegress(Phi, y)
        v = inv(Phi.' * Phi) * Phi.' * y;
    end

    %plots cos(pi*x) + cos(2pi*x)
    function plotTrueFunction()
        x = 0 : .01 : 1;
        y = cos(pi*x)+cos(2*pi*x);
        plot(x,y);
    end

    %plots dataset (x,y)
    function plotPoints(X,Y)
        plot(X, Y, 'o', 'MarkerSize', 8);
        xlabel('x'); ylabel('y');
    end

    %%%%%%%%%%%%%%%%%%%%
    % STUFF FOR PART 1 %
    %%%%%%%%%%%%%%%%%%%%
    
    % x: column vector whose entries are data points inputs x^(i)
    % y: column vector whose entries are data points outputs y^(i)
    % M: maximal degree of poly regression
    % return: w such that w(1)+w(2)x+w(3)x^2+... is the optimal polynomial
    % fit
    function v = linRegressPolynomial(x, y, M)
        Phi = zeros([length(y) M+1]);
        for i=1:length(y)
            for j=1:M+1
                Phi(i,j)=x(i)^(j-1);
            end
        end
        v = linRegress(Phi, y);
    end
    
    % w represents polynomial w(1)+w(2)x+w(3)x^2+...
    function plotPolynomial(w)
        x = 0:0.01:1;
        y = 0;
        for i = 1:length(w)
            y = y + w(i) * x.^(i-1);
        end
        plot(x,y);
    end

    %%%%%%%%%%%%%%%%%%%%
    % STUFF FOR PART 2 %
    %%%%%%%%%%%%%%%%%%%%

    %x: input
    %w: represents polynomial w(1)+w(2)*x+w(3)*x^2+...
    function v = evalPolynomial(x, w);
        ans=0;
        for i = 1:length(w)
            ans = ans + w(i)*x^(i-1);
        end
        v = ans;
    end
    
    %X: x coordinates
    %Y: y coordinates
    %w: represents polynomial w(1)+w(2)*x+w(3)*x^2+...
    %returns sum squares error of w as hypothesis for dataset (X,Y)
    function v = sumSquaresError(X,Y,w)
        squaredError = 0;
        for i = 1:length(Y)
            squaredError = squaredError + (evalPolynomial(X(i),w)-Y(i))^2;
        end
        v = squaredError;
    end
    
    %X: x coordinates
    %Y: y coordinates
    %w: represents polynomial w(1)+w(2)*x+w(3)*x^2+...
    %returns gradient of sum squares error of w as hypothesis for dataset (X,Y)
    function v = sumSquaresErrorGrad(X,Y,w)
        squaredErrorGrad = zeros([length(w) 1]);
        for i = 1:length(Y)
            tmp = zeros([length(w) 1]);
            for j = 1:length(w)
                tmp(j) = X(i)^(j-1);
            end
            squaredErrorGrad = squaredErrorGrad + 2*(evalPolynomial(X(i),w)-Y(i))*tmp;
        end
        v = squaredErrorGrad;
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

    %%%%%%%%%%%%%%%%%%%%
    % STUFF FOR PART 3 %
    %%%%%%%%%%%%%%%%%%%%

    %TODO
    
    %%%%%%%%%%%%%%%%%%%%
    % STUFF FOR PART 4 %
    %%%%%%%%%%%%%%%%%%%%

    % x: column vector whose entries are data points inputs x^(i)
    % y: column vector whose entries are data points outputs y^(i)
    % M: size of basis; basis is cos(pi*x), cos(2pi*x),...,cos(Mpi*x)
    % return: w such that w(1)*cos(pi*x)+w(2)cos(2pi*x)+... is optimal
    function v = linRegressCosines(x, y, M)
        Phi = zeros([length(y) M]);
        for i=1:length(y)
            for j=1:M
                fn = @(x) cos(j*pi*x);
                Phi(i,j)=fn(x(i));
            end
        end
        v = linRegress(Phi, y);
    end
    
    % w: coefficient vector
    % plots w(1)*cos(pi*x)+w(2)*cos(2pi*x)+...
    function plotCosines(w)
        x = 0 : .01 : 1;
        y = 0;
        for i = 1:length(w)
            fn = @(x) cos(i*pi*x);
            y = y + w(i) * fn(x);
        end
        plot(x,y);
    end
    

    %%%%%%%%%%%%%%%%%%%%
    %  IMPLEMENTATIONS %
    %%%%%%%%%%%%%%%%%%%%
    
    %does part 1 for various values M
    function part1implementation()
        [X,Y]=loadFittingDataP2(0);
        w = linRegressPolynomial(X.',Y.',0); %<-- REPLACE 0 WITH DESIRED M
        disp(w);
        hold on;
        title('Linear Regression (M=0)'); %<-- REPLACE 0 WITH DESIRED M
        plotTrueFunction();
        plotPoints(X,Y);
        plotPolynomial(w);
        hold off;
    end

    function part2implementation()
        [X,Y]=loadFittingDataP2(0);
        
        %for writeup: 
        %check that sumSquaresErrorGrad and approxSumSquaresErrorGrad
        %are the same for all w.  In particular when 
        %w = linRegressPolynomial(X.',Y.',M); <-- fill in M
        %both gradients should be 0.
        w = [1;1;1]; %<-- REPLACE THIS WITH ARBITRARY w
        
        epsilon=1e-6;
        sumSquaresFnOfW = @(ww) sumSquaresError(X,Y,ww);
        approxSumSquaresErrorGrad = approxGradient(sumSquaresFnOfW, epsilon);

        disp(sumSquaresError(X,Y,w));
        disp(sumSquaresErrorGrad(X,Y,w));
        disp(approxSumSquaresErrorGrad(w));
    end

    function part4implementation()
        [X,Y]=loadFittingDataP2(0);
        w = linRegressCosines(X.',Y.',8);
        disp(w); %<-- USE THIS TO ANSWER "how does weight vector compare to actual"
        hold on;
        title('Linear Regression with Cosine Basis')
        plotPoints(X,Y);
        plotCosines(0,1,w);
        hold off;
    end

    part2implementation();

end