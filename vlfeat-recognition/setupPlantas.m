function imdb = setupPlantas(datasetDir, varargin)
% SETUPPLANTAS   Setup the Plantas dataset
%    IMDB = SETUPPLANTAS(DATASETDIR) initializes a IMDB structure
%    representing the dataset located at DATASETDIR. The dataset
%    is supposed to have a list of training, validation and test set.
%    Also a label list file.
%
%
%    SETUPPLANTAS(..., 'OPT', VAL, ...) accepts the following
%    options:
%
%    Lite:: false
%      If set to TRUE, use at most 3 classes and at most 5 images
%      in each of TRAIN, VAL, and TEST.
%
%

% Author: Rene Octavio Queiroz Dias

% This file is a modification of the VLFeat library and is made
% available under the terms of the BSD license (see the COPYING file).

opts.lite = false;
opts = vl_argparse(opts, varargin);

% Read files
labelsFile = fopen(fullfile(datasetDir, 'labels.txt'), 'r');
trainSetFile = fopen(fullfile(datasetDir, 'train.txt'), 'r');
validSetFile = fopen(fullfile(datasetDir, 'val.txt'), 'r');
testSetFile = fopen(fullfile(datasetDir, 'test.txt'), 'r');

% Construct image database imdb structure
imdb.meta.sets = {'train', 'val', 'test'} ;

% Read labels
names = {};
tline = fgetl(labelsFile);
while ischar(tline)
    names{end+1} = tline;
    tline = fgetl(labelsFile);
end
imdb.meta.classes = names;

names = {};
classes = {};
imagesPath = {};
sets = [];

imageDir = '';

% Read Training Set
tline = fgetl(trainSetFile);
while ischar(tline)
    tline = strsplit(tline);
    class = str2double(tline(end)) + 1;
    tline = strjoin(tline(1:end-1));

    imagesPath{end+1} = tline;

    [pathstr, name, ext] = fileparts(tline);

    className = strsplit(pathstr, filesep);

    if isempty(imageDir)
        imageDir = strjoin(className(1:end-1), filesep);
    end

    className = className(end);

    name = strcat(className, filesep, name, ext);

    names{end+1} = name;
    classes{end+1} = class;
    sets(end+1) = 1;

    tline = fgetl(trainSetFile);
end

% Read Validation Set
tline = fgetl(validSetFile);
while ischar(tline)
    tline = strsplit(tline);
    class = str2double(tline(end)) + 1;
    tline = strjoin(tline(1:end-1));

    imagesPath{end+1} = tline;

    [pathstr, name, ext] = fileparts(tline);

    className = strsplit(pathstr, filesep);
    className = className(end);

    name = strcat(className, filesep, name, ext);
    names{end+1} = name;
    classes{end+1} = class;
    sets(end+1) = 2;

    tline = fgetl(validSetFile);
end

% Read Test Set
tline = fgetl(testSetFile);
while ischar(tline)
    tline = strsplit(tline);
    class = str2double(tline(end)) + 1;
    tline = strjoin(tline(1:end-1));

    imagesPath{end+1} = tline;

    [pathstr, name, ext] = fileparts(tline);

    className = strsplit(pathstr, filesep);
    className = className(end);

    name = strcat(className, filesep, name, ext);
    names{end+1} = name;
    classes{end+1} = class;
    sets(end+1) = 3;

    tline = fgetl(testSetFile);
end

names = cat(2,names{:});
classes = cat(2,classes{:});
ids = 1:numel(names);

imdb.images.id = ids;
imdb.images.name = names;
imdb.images.set = sets;
imdb.images.class = classes;
imdb.images.imagesPath = imagesPath;
imdb.imageDir = imageDir;

if opts.lite
  ok = {} ;
  for c = 1:3
    ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 1), 5) ;
    ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 2), 5) ;
    ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 3), 5) ;
  end
  ok = cat(2, ok{:}) ;
  imdb.meta.classes = imdb.meta.classes(1:3) ;
  imdb.images.id = imdb.images.id(ok) ;
  imdb.images.name = imdb.images.name(ok) ;
  imdb.images.set = imdb.images.set(ok) ;
  imdb.images.class = imdb.images.class(ok) ;
  imdb.images.imagesPath = imdb.images.imagesPath(ok);
end
