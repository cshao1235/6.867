%x = logspace(5,8.5);

X = [3e5; 1e6; 3e6; 1e7; 3e7; 1e8];
Y = [159, 45, 13, 14, 522, 1];

set(gca,'xscale','log');


loglog(X, Y, 'o', 'MarkerSize', 8);
    xlabel('x'); ylabel('y');

    % plot(3e5,159);
% plot(1e6,45);
% plot(3e6,13);
% plot(1e7,14);
% plot(3e7,522);
% plot(1e8,1);

%loglog(x,exp(x),'-s')
grid on