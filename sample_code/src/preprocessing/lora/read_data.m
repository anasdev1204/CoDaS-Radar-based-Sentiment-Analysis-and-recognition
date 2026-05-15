function [signal] = read_file2(path, fn, N)
%READ_FILE: read data from a file and return
%   path: path to the file
%   fn: name of the file
%   N: number of the antenna (default: 1)

rawData = cell(1, N); % N = 1
shortestLen = 1e10;

for i = 1:N
    % get the full file path
	fileName = fullfile(path,fn)
    % output running log
    disp(['Trying to open file: ', fileName]);
    
    % open file
    file = fopen(fileName, 'rb');
    if file == -1
        error(['Cannot open file: ', fileName]);
    end
    
    % read file
    rawData{i} = fread(file, 'float32');
    
    % update the shortest length
    shortestLen = min(shortestLen, length(rawData{i}));
    
    % close file
    fclose(file);
end

signal = zeros(shortestLen/2, N);

for i = 1:N
    % obtain data and extract the data with the shortest length
    tempData = rawData{i}(1:shortestLen);
    
    % extract the real part and the imaginary part
    is = tempData(1:2:end);
    qs = tempData(2:2:end);
    
    % combine into a complex signal
    signal(:,i) = complex(is, qs);
end