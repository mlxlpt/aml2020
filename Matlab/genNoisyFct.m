N = 1000;

CENTER_DATASET = false;

NOISE_VAR = 0.2;

x_min = -10;
x_max = 20;

X = x_min + rand(N,1)*(x_max-x_min);
X = sort(X);

%X = zeros(N,1);
Y=X;

f1 = @(x) x+3*sin(x*1);
f2 = @(x) x+1.5*sin(x*3);
f3 = @(x) x+3*sin(x*1);

for i=1:N %j'aurais pu le faire sans boucle...
    if(i<350)
        Y(i) = f1(X(i))+randn*sqrt(NOISE_VAR);
    elseif(i<650)
        Y(i) = f2(X(i))+randn*sqrt(NOISE_VAR);
    else
        Y(i) = f3(X(i))+randn*sqrt(NOISE_VAR);
    end
end

if CENTER_DATASET
    X_c = X-mean(X);
else
    X_c = X;
end

figure();
scatter(X_c,Y)

data = [X,Y]; %X, NOISY VALUE, TRUE VALUE
dlmwrite('genfct.csv',data,'delimiter',',','precision', 6)