close all;
clear all;
%B = dlmread('shalegas.csv', ',', 1,0);
B = readtable('shalegas_data.csv');



for i=1:size(B,2)
    for j=i+1:size(B,2)
        figure
        if (i == 1)
           X=zeros(size(B,1),1);
            for k=1:size(B,1)
                temp=datenum(B.(B.Properties.VariableNames{1})(k));
                X(k) = addtodate(temp, 2000, 'year');
            end
        else
            X=sort(B.(B.Properties.VariableNames{i}));
        end
        plot(X,B.(B.Properties.VariableNames{j}));
        xlabel(B.Properties.VariableNames{i});
        ylabel(B.Properties.VariableNames{j});
    end
    fprintf('%d %d\n', i,j)
    pause
    close all;
end