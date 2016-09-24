function gradientDescent()

function w = gradientDescent_1(fn, grad, startPoint, stepSize, convergenceThreshold)
currentPoint = startPoint;
while (norm(grad(currentPoint)) > convergenceThreshold)
    currentPoint = currentPoint - stepSize*grad(currentPoint);
end
w = currentPoint;
end

function v = mymvnpdf(mean, cov)
n = size(cov, 1);
    function w = out(x)
        w = -1/sqrt((2*pi)^n*det(cov))*exp(-1/2*(x - mean).'*inv(cov)*(x - mean));
    end
v = @(x) out(x);
end

function v = mymvnpdfgrad(mean, cov)
    function w = out(x)
        f = mymvnpdf(mean, cov);
        w = -f(x)*inv(cov)*(x - mean);
    end
v = @(x) out(x);
end

[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();
myfn = mymvnpdf(gaussMean, gaussCov);
myfngrad = mymvnpdfgrad(gaussMean, gaussCov);
disp(mvnpdf([9.9; 10.1], gaussMean, gaussCov));
disp(myfn([9.9; 10.1]));
startPoint = [9.9; 10.1];
stepSize = 0.00001;
convergenceThreshold = 0.00001;
v = gradientDescent_1(mymvnpdf(gaussMean, gaussCov), mymvnpdfgrad(gaussMean, gaussCov), startPoint, stepSize, convergenceThreshold)

end