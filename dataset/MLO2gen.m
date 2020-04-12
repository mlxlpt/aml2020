A = readmatrix('./old_datasets/MLO_data.csv');

A(A==-1) = NaN; % negative values are NaN according to site

ACTIVATE_NOISE = false;

X=A(:,3);
Y=A(:,4);

%estimate mean step between each date record
d=0;
for i=1:(length(X)-1)
   d = d + X(i+1)-X(i);
end
d=d/(length(X)-1);

N_new = length(X)
X_new = zeros(N_new,1);
X_new(1) = X(end)+d;
for i=2:N_new
   X_new(i) = round(X_new(i-1)+d*(1+sin(rand*2*pi)/50),3); 
end
Y_new = Y+Y(end)-Y(1);
f = @(x) Y(end)+(x-X(end))*(Y(end)-Y(1))/(X(end)-X(1));
for i=1:N_new
   Y_new(i) = 2*f(X_new(i))-Y_new(i);
end

Xf = [X;X_new];
Yf = [Y;Y_new];

figure;
scatter([X;X_new],[Y;Y_new]);
hold on;
fact = -2*pi/length(Yf); % sin factor to augment curvature
T=[0:1/(length(Yf)-1):1];

for i=1:length(Yf)
    rn = 3.5*randn;
    if (abs(rn) > 20)
        rn=20*sign(rn);
        rn=rn*ACTIVATE_NOISE;
    end
    Yf(i) = Yf(i)*(1+1/(30+rn)*sin(pi/6+fact*i))-0.75*(i^(1/4))*log(i);
end

scatter(Xf,Yf);
data = [Xf,Yf];
dlmwrite('test.csv',data,'delimiter',',','precision', 9)