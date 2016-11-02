%#3
label = zeros(size(datar,1),1,'uint8');
for i = 1:10
    label((i-1)*1500+1:i*1500,1)=uint8(i-1);
end
