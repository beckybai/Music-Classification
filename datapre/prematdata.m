
% #1 
path = 'D:\junior\artificialnn\musicwave\country';
%先观察这个时间区段内，一张图片的长度
msize =33*2500;
time = 1:2500;
Musics = dir(fullfile(path));
fragments = length(Musics);

num = 1;
name =  strcat(path,'\',Musics(10).name);
[spectrom,s2] = example(name);
spectrom = spectrom(:,time);
data = zeros(1000,msize,'double'); 
for i = 1:fragments
    if Musics(i).isdir ==0
        oldname = Musics(i).name;
        flag = regexp(oldname,'.0');
        names1 = oldname(1:flag(1,1)-1);
        name =  strcat(path,'\',Musics(i).name);
        spectrom = example(name);
        spectrom = spectrom(:,time);
        data(num,:) = reshape(spectrom,1,msize);
        num = num + 1;
    end
end