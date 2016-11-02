clear;
% #1 
path = 'D:\Files for Courses 5\Artificial Neural Network\Final_project\country';
%先观察这个时间区段内，一张图片的长度
msize =1025;
pieces=600;
time = 1:pieces;
Musics = dir(fullfile(path));
fragments = length(Musics);

num = 1;
name =  strcat(path,'\',Musics(10).name);
[spectrum,s2] = example(name);
spectrum = spectrum(:,time);
data = zeros(1000,msize,pieces,'single'); 
for i = 1:fragments
    if Musics(i).isdir ==0
        oldname = Musics(i).name;
        flag = regexp(oldname,'.0');
        names1 = oldname(1:flag(1,1)-1);
        name =  strcat(path,'\',Musics(i).name);
        spectrum = example(name);
        spectrum = spectrum(:,time);
        
        data(num,:,:) = spectrum;
        % data(num,:,:) = reshape(spectrum,1,msize);
        fprintf('Files pocessed:%d\n',num);
        num = num + 1;
        
    end
end


datac = zeros(size(data,1),72,size(data,3));
datac(:,1:16,:) =data(:,1:16,:);

for i=4:9
   for j = 1:8
       datac(:,-8+(i-1)*8+j,:) = mean(data(:,2^i+(j-1)*2^(i-3)+1:2^i+(j)*2^(i-3),:),2);
   end
end

len = 600;
stack = 30;
wav_len = floor(len/stack); 
%datar = datac(:,:,1:wav_len*stack);
%datar = reshape(datar,1000,33,wav_len,30);
py_data =zeros(1000*stack,72,wav_len,'single');
for i=1:1000*stack
    j = mod(i-1,stack)+1;
    py_data(i,:,:) = datac(ceil(i/stack),:,((j-1)*wav_len+1:j*wav_len));
end

label = zeros(size(py_data,1),1,'single');
for i = 1:10
    label((i-1)*stack*100+1:i*stack*100,1)=uint8(i-1);
end

C=randperm(stack*1000);
py_data = py_data-mean(mean(mean(py_data)));
% for i=1:10000
%     py_data(i,:,:) = py_data(i,:,:)/(mean(var(py_data(i,:,:),0,2)))^0.5;
% end
py_data=py_data(C,:,:);
label=label(C);

save Train_y5 label
save Train_x5 py_data




