path = 'D:\junior\artificialnn\musicwave\country';
Musics = dir(fullfile(path));
fragments = length(Musics);
class = 100;
num = 1;
name =  strcat(path,'\',Musics(10).name);
[spectrom,s2] = example(name);
data = zeros(size(spectrom,1)*size(spectrom,2),100,'uint8'); 
for i = 1:fragments
    if Musics(i).isdir ==0
        oldname = Musics(i).name;
        flag = regexp(oldname,'.0');
        names1 = oldname(1:flag(1,1)-1);
        name =  strcat(path,'\',Musics(i).name);
        spectrom = example(name);
        data(:,num) = reshape(spectrom,size(spectrom,1)*size(spectrom,2),1);
        if(mod(num,100)==0)
            filename = strcat(names1,'c2.bin');
            fileID = fopen(filename,'a');
            num = 1;
            fwrite (fileID,data,'uint8');
            fclose(fileID);
            continue
        end
        num = num + 1;
    end
end