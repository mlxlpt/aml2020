N = 30;

CENTER_DATASET = false;

PULSATION = 0.5;
AMPLITUDE_SINE = 1;
PHASE = pi/2;

NOISE_VAR = 0.02

SINE_MIN_ARG = 0;
SINE_MAX_ARG = 3*pi;

x_min = SINE_MIN_ARG/PULSATION - 0.5;
x_max = SINE_MAX_ARG/PULSATION + 0.5;

X = x_min + rand(N,1)*(x_max-x_min);
X = sort(X);

%X = zeros(N,1);
Y=X;

f = @(x) AMPLITUDE_SINE*sin(x*PULSATION + PHASE);

for i=1:N %j'aurais pu le faire sans boucle...
    Y(i) = f(X(i))+randn*sqrt(NOISE_VAR);
end

if CENTER_DATASET
    X_c = X-mean(X);
else
    X_c = X;
end

X_plot = [X_c(1):0.001:X_c(end)]';

figure();
plot(X_plot',f(X_plot)');
hold on;
scatter(X_c,Y)
grid on;

data = [X_c,Y,f(X)]; %X, NOISY VALUE, TRUE VALUE
dlmwrite('noisysine.csv',data,'delimiter',',','precision', 6)