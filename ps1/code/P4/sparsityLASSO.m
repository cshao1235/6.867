function sparsityLASSO()

    function part1Implementation()
        [x, y] = lassoTestData()
        n = length(y);
        M = 13;
        X = zeros(n, M);
        for r = 1:n
            X(r, 1) = x(r);
            for c = 2:M
                X(r, c) = sin(0.4*pi*x(r)*(c - 1));
            end
        end
        l = 5;
        lambda = zeros(l);
        for r = 1:l
            lambda(r) = 10.0^(-r);
        end
        w = lasso(X, y, 'Lambda', lambda);
        lassoPlot(w);
    end

    part1Implementation();
end