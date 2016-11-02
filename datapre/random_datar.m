
%# 3
da  = datar - repmat(mean(datar),size(datar,1),1);
traindouble = zeros(12000,size(da,2));
testdouble = zeros(3000,size(da,2));
for i = 1:10
    traindouble((i-1)*1200+1:i*1200,:) = da((i-1)*1500+1:(i-1)*1500+1200,:);
    trainl((i-1)*1200+1:i*1200,:) = label((i-1)*1500+1:(i-1)*1500+1200,:);
    
    testdouble((i-1)*300+1:i*300,:) = da((i-1)*1500+1201:(i)*1500,:);
    testl((i-1)*300+1:i*300,:) = label((i-1)*1500+1201:(i)*1500,:);

end

x = randperm(12000);
traindouble = traindouble(x,:);
trainl = trainl(x);



save trmint2 traindouble trainl
save temint2 testdouble testl